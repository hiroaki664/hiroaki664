from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import os
import random

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # will fall back to PIL if available

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

#-------------------------------------------------------------------------
# オプションの主な引数
#
# labels="folder" | "csv" | "none"：ラベル付与方法
#
# labels_csv="path/to/labels.csv"＋filename_col/label_col：CSVマップ対応
#
# grayscale=True/False/None：モード固定 or 自動
#
# resize=(W, H)：リサイズ
#
# normalize=True：整数画像を[0,1]に正規化
#
# channels_first=True：PyTorch（CHW）向け
#
# split(train,val,test, stratify=True)：層化分割対応
#
# iter_batches(batch_size=..., idxs=...)：軽量バッチイテレータ
#-------------------------------------------------------------------------

# ---------------------------
# Low-level image I/O helpers
# ---------------------------

def _imread(path: Path, grayscale: Optional[bool] = None) -> np.ndarray:
    """
    Read an image as np.ndarray (H, W, C) in RGB (or (H, W) for grayscale).
    Tries multiple backends to avoid Unicode/OneDrive path issues on Windows.
    Order: OpenCV (imdecode) → OpenCV (imread) → PIL.
    """
    # 1) Try OpenCV via imdecode to better handle Unicode/long paths
    if cv2 is not None:
        try:
            flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
            data = np.fromfile(str(path), dtype=np.uint8)  # works with Unicode paths on Windows
            arr = cv2.imdecode(data, flags)
            if arr is not None:
                if not grayscale and arr.ndim == 3:
                    if arr.shape[2] == 3:
                        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    elif arr.shape[2] == 4:
                        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
                return arr
        except Exception:
            pass  # fall through to other methods

        # 2) Fallback to OpenCV imread (may fail on some Unicode paths)
        try:
            flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
            arr = cv2.imread(str(path), flags)
            if arr is not None:
                if not grayscale and arr.ndim == 3:
                    if arr.shape[2] == 3:
                        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    elif arr.shape[2] == 4:
                        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
                return arr
        except Exception:
            pass

    # 3) PIL fallback
    if Image is not None:
        try:
            img = Image.open(path)
            if grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGBA") if img.mode in {"RGBA", "LA"} else img.convert("RGB")
            return np.array(img)
        except Exception:
            pass

    raise FileNotFoundError(f"Failed to read image: {path}")


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure image has shape (H, W, C) where C>=1. For grayscale, add channel dimension."""
    if arr.ndim == 2:
        return arr[:, :, None]
    return arr


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class ImageRecord:
    path: Path
    label: Optional[Union[str, int]] = None
    id: Optional[str] = None  # e.g., filename stem or any external id


# ---------------------------
# Main loader
# ---------------------------

class ImageFolderLoader:
    """
    A lightweight, framework-agnostic image dataset loader.

    Features:
    - Scan one or multiple root folders for images.
    - Labels from folder names (root/class_x/xxx.png) or CSV mapping.
    - Optional transforms, resizing, dtype/normalization, grayscale.
    - Split into train/val/test with a fixed seed for reproducibility.
    - Simple batch iterator.
    - Optional conversion to a PyTorch Dataset (without forcing torch dependency).

    Example folder structure (folder-as-label):
        root/
          ├─ class_a/ img1.png, img2.png
          └─ class_b/ img3.png

    Alternatively, provide a CSV mapping with columns [filename, label].
    """

    IMAGE_PATTERNS: Tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.pgm")

    def __init__(
        self,
        roots: Union[str, Path, Sequence[Union[str, Path]]],
        patterns: Sequence[str] | None = None,
        recursive: bool = True,
        labels: str = "folder",  # 'folder' | 'csv' | 'none'
        labels_csv: Optional[Union[str, Path]] = None,
        filename_col: str = "filename",
        label_col: str = "label",
        grayscale: Optional[bool] = None,
        resize: Optional[Tuple[int, int]] = None,  # (W, H)
        normalize: bool = True,
        dtype: str = "float32",
        channels_first: bool = False,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        id_from: str = "stem",  # 'stem' | 'name' | 'relative'
    ):
        self.roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
        self.patterns = tuple(patterns) if patterns else self.IMAGE_PATTERNS
        self.recursive = recursive
        self.labels_mode = labels
        self.labels_csv = Path(labels_csv) if labels_csv else None
        self.filename_col = filename_col
        self.label_col = label_col
        self.grayscale = grayscale
        self.resize = resize
        self.normalize = normalize
        self.dtype = dtype
        self.channels_first = channels_first
        self.transform = transform
        self.id_from = id_from

        self.records: List[ImageRecord] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._build_index()

    # --------- public API ---------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        arr = _imread(rec.path, grayscale=self.grayscale)

        if self.resize is not None:
            w, h = self.resize
            if cv2 is not None:
                interp = cv2.INTER_AREA if (arr.shape[0] >= h and arr.shape[1] >= w) else cv2.INTER_LINEAR
                arr = cv2.resize(arr, (w, h), interpolation=interp)
            elif Image is not None:
                img = Image.fromarray(arr)
                img = img.resize((w, h))
                arr = np.array(img)

        arr = _ensure_3d(arr)

        if self.normalize:
            # handle integer images of various bit-depths; scale to [0,1]
            info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
            if info is not None:
                arr = arr.astype(np.float32) / float(info.max)
            else:
                arr = arr.astype(np.float32)

        if self.transform is not None:
            arr = self.transform(arr)

        if self.channels_first:
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

        sample = {
            "image": arr.astype(self.dtype),
            "path": str(rec.path),
            "id": rec.id,
            "label": rec.label,
            "label_idx": self._label_to_index(rec.label) if rec.label is not None else None,
        }
        return sample

    def get_labels(self) -> Optional[List[Union[str, int]]]:
        labels = [r.label for r in self.records]
        return labels if any(l is not None for l in labels) else None

    def class_mapping(self) -> Dict[int, str]:
        return dict(self.idx_to_class)

    def split(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 42,
        stratify: bool = True,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Return index splits (train_idx, val_idx, test_idx)."""
        assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
        n = len(self)
        rng = random.Random(seed)

        if stratify and self.get_labels() is not None:
            # group indices by class for stratified split
            by_class: Dict[Any, List[int]] = {}
            for i, r in enumerate(self.records):
                by_class.setdefault(r.label, []).append(i)

            train_idx: List[int] = []
            val_idx: List[int] = []
            test_idx: List[int] = []
            for _, idxs in by_class.items():
                rng.shuffle(idxs)
                n_c = len(idxs)
                n_train = int(round(n_c * train))
                n_val = int(round(n_c * val))
                n_test = n_c - n_train - n_val
                train_idx.extend(idxs[:n_train])
                val_idx.extend(idxs[n_train:n_train + n_val])
                test_idx.extend(idxs[n_train + n_val:])
            return train_idx, val_idx, test_idx
        else:
            idxs = list(range(n))
            rng.shuffle(idxs)
            n_train = int(round(n * train))
            n_val = int(round(n * val))
            n_test = n - n_train - n_val
            return idxs[:n_train], idxs[n_train:n_train + n_val], idxs[n_train + n_val:]

    def iter_batches(
        self, batch_size: int, idxs: Optional[Sequence[int]] = None, shuffle: bool = True, seed: int = 0
    ) -> Iterable[List[Dict[str, Any]]]:
        idxs = list(range(len(self))) if idxs is None else list(idxs)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            yield [self[j] for j in idxs[i:i + batch_size]]

    def to_torch_dataset(self):
        """Return a PyTorch Dataset wrapping this loader without forcing torch dependency here."""
        try:
            import torch  # type: ignore
            from torch.utils.data import Dataset  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("PyTorch is not installed. Install torch to use to_torch_dataset().") from e

        outer = self

        class _TorchDataset(Dataset):
            def __len__(self):
                return len(outer)

            def __getitem__(self, idx: int):
                s = outer[idx]
                x = torch.from_numpy(s["image"])  # type: ignore
                y = s["label_idx"]
                if y is None:
                    return x, s
                return x, int(y)

        return _TorchDataset()

    # --------- index building ---------

    def _build_index(self):
        files: List[Path] = []
        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(f"Root not found: {root}")
            if self.recursive:
                for pat in self.patterns:
                    files.extend(root.rglob(pat))
            else:
                for pat in self.patterns:
                    files.extend(root.glob(pat))
        files = sorted(set(files))
        if len(files) == 0:
            raise FileNotFoundError(f"No images found under {self.roots} with patterns {self.patterns}")

        if self.labels_mode == "csv":
            import pandas as pd  # local import
            if self.labels_csv is None or not self.labels_csv.exists():
                raise FileNotFoundError("labels='csv' but labels_csv not provided or missing.")
            df = pd.read_csv(self.labels_csv)
            mapping: Dict[str, Any] = {str(k): v for k, v in zip(df[self.filename_col], df[self.label_col])}

            for p in files:
                key = p.name  # match by filename
                label = mapping.get(key)
                rec = ImageRecord(path=p, label=label, id=self._make_id(p))
                self.records.append(rec)

        elif self.labels_mode == "folder":
            # label = immediate parent folder name relative to the first matching root
            for p in files:
                # choose the closest root ancestor
                root = max((r for r in self.roots if r in p.parents), key=lambda r: len(str(r)), default=None)
                label = p.parent.name if root is not None and p.parent != root else None
                rec = ImageRecord(path=p, label=label, id=self._make_id(p))
                self.records.append(rec)
        elif self.labels_mode == "none":
            for p in files:
                self.records.append(ImageRecord(path=p, label=None, id=self._make_id(p)))
        else:
            raise ValueError("labels must be one of: 'folder', 'csv', 'none'")

        # Build class mappings if labels present
        labels = [r.label for r in self.records if r.label is not None]
        if labels:
            unique = sorted(set(str(l) for l in labels))
            self.class_to_idx = {c: i for i, c in enumerate(unique)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def _make_id(self, p: Path) -> str:
        if self.id_from == "name":
            return p.name
        if self.id_from == "relative":
            # relative to the first root that is a parent
            for r in self.roots:
                if r in p.parents:
                    return str(p.relative_to(r))
            return p.name
        return p.stem

    def _label_to_index(self, label: Optional[Union[str, int]]) -> Optional[int]:
        if label is None:
            return None
        key = str(label)
        if key not in self.class_to_idx:
            # unseen label (possible when mixing csv/folder)
            self.class_to_idx[key] = len(self.class_to_idx)
            self.idx_to_class[len(self.idx_to_class)] = key
        return self.class_to_idx[key]


# ---------------------------
# Quick usage examples
# ---------------------------
if __name__ == "__main__":
    # Adjust these paths to your environment (examples based on your previous messages)
    ROOT_CLSM = r"C:\Users\oyaji\OneDrive - Kyushu University\デスクトップ\nanoparticle\clsm"
    ROOT_SEM  = r"C:\Users\oyaji\OneDrive - Kyushu University\デスクトップ\nanoparticle\sem"

    loader = ImageFolderLoader(
        roots=[ROOT_CLSM, ROOT_SEM],
        labels="folder",         # use parent folder as label
        grayscale=None,           # auto; set True to force grayscale
        resize=None,              # e.g., (256, 256)
        normalize=True,           # scale to [0,1]
        dtype="float32",
        channels_first=False,     # set True for PyTorch CHW
        transform=None,           # or provide a callable that maps np.ndarray -> np.ndarray
        id_from="stem",
    )

    print(f"Total images: {len(loader)}")
    if loader.get_labels() is not None:
        print("Classes:", loader.class_mapping())

    # Fetch one sample
    s0 = loader[0]
    print("Sample keys:", s0.keys())
    print("Image shape:", s0["image"].shape, "label:", s0["label"], "path:", s0["path"]) 

    # Create a torch dataset if PyTorch is installed
    try:
        ds = loader.to_torch_dataset()
        print("Torch dataset length:", len(ds))
    except ImportError:
        print("PyTorch not installed; skipping torch dataset demo.")

    # Split and iterate mini-batches
    tr, va, te = loader.split(train=0.7, val=0.15, test=0.15, seed=42)
    print(len(tr), len(va), len(te))

    for batch in loader.iter_batches(batch_size=4, idxs=tr, shuffle=True, seed=0):
        # batch is a list of dicts {image, path, id, label, label_idx}
        print("Batch size:", len(batch))
        break