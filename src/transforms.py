from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # some ops will raise if cv2 is missing

# ============================================================
# Utilities
# ============================================================

def _ensure_hwc(x: np.ndarray) -> np.ndarray:
    """Ensure (H,W,C). If grayscale (H,W), add last channel."""
    if x.ndim == 2:
        x = x[:, :, None]
    return x


def _to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float32)
    if x.max() <= 1.0:
        x = (x * 255.0).clip(0, 255)
    return x.astype(np.uint8)


def _to_float01(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32 or x.dtype == np.float64:
        m = x.max()
        if m > 1.0:
            return (x / 255.0).astype(np.float32)
        return x.astype(np.float32)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        return (x.astype(np.float32) / float(info.max))
    return x.astype(np.float32)


# ============================================================
# Combinators
# ============================================================

def compose(*funcs: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Chain multiple transforms left→right.

    Example:
        t = compose(to_gray, resize((256,256)), clahe(), normalize01())
        y = t(x)
    """
    def _apply(x: np.ndarray) -> np.ndarray:
        for f in funcs:
            x = f(x)
        return x
    return _apply


class RandomApply:
    """Apply a transform with probability p (for data augmentation)."""
    def __init__(self, transform: Callable[[np.ndarray], np.ndarray], p: float = 0.5):
        self.t = transform
        self.p = float(p)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return self.t(x)
        return x


# ============================================================
# Basic conversions / normalization
# ============================================================

def to_gray() -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        x = _ensure_hwc(x)
        if x.shape[2] == 1:
            return x
        if cv2 is None:
            # naive RGB→Gray weights
            rgb = x.astype(np.float32)
            g = (0.2989*rgb[...,0] + 0.5870*rgb[...,1] + 0.1140*rgb[...,2]).astype(rgb.dtype)
            return g[...,None]
        u8 = _to_uint8(x)
        g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
        return g[..., None]
    return _f


def normalize01() -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        return _to_float01(x)
    return _f


def normalize_minmax(new_min: float = 0.0, new_max: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        xmin, xmax = float(x.min()), float(x.max())
        if xmax <= xmin + 1e-12:
            return np.full_like(x, new_min, dtype=np.float32)
        x = (x - xmin) / (xmax - xmin)
        return (x * (new_max - new_min) + new_min).astype(np.float32)
    return _f


# ============================================================
# Geometric transforms
# ============================================================

def resize(size: Tuple[int,int]) -> Callable[[np.ndarray], np.ndarray]:
    W, H = size
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for resize")
        x = _ensure_hwc(x)
        u8 = _to_uint8(x)
        interp = cv2.INTER_AREA if (u8.shape[0] >= H and u8.shape[1] >= W) else cv2.INTER_LINEAR
        y = cv2.resize(u8, (W, H), interpolation=interp)
        return _ensure_hwc(y)
    return _f


def center_crop(size: Tuple[int,int]) -> Callable[[np.ndarray], np.ndarray]:
    W, H = size
    def _f(x: np.ndarray) -> np.ndarray:
        x = _ensure_hwc(x)
        h, w = x.shape[:2]
        cx, cy = w//2, h//2
        x0 = max(0, cx - W//2)
        y0 = max(0, cy - H//2)
        x1, y1 = min(w, x0 + W), min(h, y0 + H)
        crop = x[y0:y1, x0:x1]
        # pad if needed
        pad_h = H - crop.shape[0]
        pad_w = W - crop.shape[1]
        if pad_h>0 or pad_w>0:
            crop = np.pad(crop, ((0,pad_h),(0,pad_w),(0,0)), mode='edge')
        return crop
    return _f


def pad_to_square() -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        x = _ensure_hwc(x)
        h, w = x.shape[:2]
        m = max(h, w)
        pad_h = m - h
        pad_w = m - w
        return np.pad(x, ((0,pad_h),(0,pad_w),(0,0)), mode='edge')
    return _f


def rotate(angle_deg: float, center: Optional[Tuple[float,float]] = None) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for rotate")
        x = _ensure_hwc(x)
        h, w = x.shape[:2]
        c = center if center is not None else (w/2.0, h/2.0)
        M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
        u8 = _to_uint8(x)
        y = cv2.warpAffine(u8, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return _ensure_hwc(y)
    return _f


def hflip() -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(_ensure_hwc(x)[:, ::-1, :])
    return _f


def vflip() -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(_ensure_hwc(x)[::-1, :, :])
    return _f


# ============================================================
# Photometric transforms
# ============================================================

def gaussian_blur(ksize: Tuple[int,int] = (3,3), sigma: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for gaussian_blur")
        u8 = _to_uint8(_ensure_hwc(x))
        y = cv2.GaussianBlur(u8, ksize, sigma)
        return _ensure_hwc(y)
    return _f


def median_blur(ksize: int = 3) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for median_blur")
        u8 = _to_uint8(_ensure_hwc(x))
        y = cv2.medianBlur(u8, ksize)
        return _ensure_hwc(y)
    return _f


def bilateral_denoise(d: int = 5, sigma_color: float = 75, sigma_space: float = 75) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for bilateral_denoise")
        u8 = _to_uint8(_ensure_hwc(x))
        y = cv2.bilateralFilter(u8, d, sigma_color, sigma_space)
        return _ensure_hwc(y)
    return _f


def clahe(clip_limit: float = 2.0, tile_grid_size: Tuple[int,int] = (8,8)) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for clahe")
        u8 = _to_uint8(_ensure_hwc(x))
        c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # --- 1ch（グレー）にも対応 ---
        if u8.shape[2] == 1:
            l2 = c.apply(u8[..., 0])
            return l2[..., None]
        # --- 3ch（RGB）は L チャンネルに適用 ---
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = c.apply(l)
        lab2 = cv2.merge((l2, a, b))
        y = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return _ensure_hwc(y)
    return _f


def hist_equalize() -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for hist_equalize")
        u8 = _to_uint8(_ensure_hwc(x))
        if u8.shape[2] == 1:
            g = cv2.equalizeHist(u8[...,0])
            return g[...,None]
        y = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
        y0, Cr, Cb = cv2.split(y)
        y1 = cv2.equalizeHist(y0)
        y2 = cv2.merge((y1, Cr, Cb))
        out = cv2.cvtColor(y2, cv2.COLOR_YCrCb2RGB)
        return _ensure_hwc(out)
    return _f


# ============================================================
# Thresholding / morphology / edges
# ============================================================

def otsu_threshold() -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for otsu_threshold")
        u8 = _to_uint8(_ensure_hwc(x))
        if u8.shape[2] != 1:
            u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)[...,None]
        _, th = cv2.threshold(u8[...,0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th[...,None]
    return _f


def adaptive_threshold(block_size: int = 11, C: int = 2) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for adaptive_threshold")
        u8 = _to_uint8(_ensure_hwc(x))
        if u8.shape[2] != 1:
            u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)[...,None]
        th = cv2.adaptiveThreshold(u8[...,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, C)
        return th[...,None]
    return _f


def morph_open(ksize: Tuple[int,int]=(3,3), iterations: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for morph_open")
        u8 = _to_uint8(_ensure_hwc(x))
        kernel = np.ones(ksize, np.uint8)
        y = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel, iterations=iterations)
        return _ensure_hwc(y)
    return _f


def morph_close(ksize: Tuple[int,int]=(3,3), iterations: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for morph_close")
        u8 = _to_uint8(_ensure_hwc(x))
        kernel = np.ones(ksize, np.uint8)
        y = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return _ensure_hwc(y)
    return _f


def canny_edges(th1: int = 100, th2: int = 200) -> Callable[[np.ndarray], np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required for canny_edges")
        u8 = _to_uint8(_ensure_hwc(x))
        if u8.shape[2] != 1:
            u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)[...,None]
        e = cv2.Canny(u8[...,0], th1, th2)
        return e[...,None]
    return _f


# ============================================================
# Example (manual test)
# ============================================================
if __name__ == "__main__":
    # quick self-test with random image
    img = (np.random.rand(128, 256, 3) * 255).astype(np.uint8)
    t = compose(
        pad_to_square(),
        resize((256,256)),
        gaussian_blur((3,3)),
        clahe(),
        normalize01(),
    )
    out = t(img)
    print("in:", img.shape, img.dtype, "out:", out.shape, out.dtype, out.min(), out.max())
