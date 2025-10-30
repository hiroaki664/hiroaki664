
"""
preprocess_utils.py
-------------------
よく使う画像前処理（特にSEM向け）を一つにまとめたユーティリティクラス。

依存:
- numpy
- opencv-python (cv2)
- Pillow (保存時に便利だが必須ではない)

使い方例:
    from preprocess_utils import ImgPreproc as IP
    import cv2

    rgb = IP.imread_any("sample.png")
    g   = IP.to_u8_gray(rgb)
    gff = IP.flatfield(g, sigma=200)
    gbp = IP.dog(gff, sigma_low=2, sigma_high=150)
    den = IP.nlm(gbp, h=10)
    th  = IP.otsu(den)
    mask= IP.morph_and_fill(th)
    mask= IP.filter_by_shape(mask, diam_range_px=(60,160))
    overlay = IP.overlay(rgb, mask)
"""

from typing import Tuple, Optional
import numpy as np
import cv2

class ImgPreproc:
    # ========== 入出力ユーティリティ ==========
    @staticmethod
    def imread_any(path: str) -> np.ndarray:
        """拡張子に依存せず画像をRGBとして読み込む。"""
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img[..., :3][:, :, ::-1]
        return rgb

    @staticmethod
    def to_u8_gray(rgb: np.ndarray) -> np.ndarray:
        """RGB→uint8グレースケール（[0,255]）。"""
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if g.dtype != np.uint8:
            if np.issubdtype(g.dtype, np.integer):
                info = np.iinfo(g.dtype)
                g = (g.astype(np.float32) / info.max * 255).clip(0,255).astype(np.uint8)
            else:
                g = (g.clip(0,1) * 255).astype(np.uint8)
        return g

    # ========== スケールバー除去 ==========
    @staticmethod
    def remove_scale_bar(rgb: np.ndarray, frac: float = 0.11, inpaint_radius: int = 3):
        """画像下端の一定割合をinpaintで埋め戻す。"""
        h, w = rgb.shape[:2]
        band_h = int(h * float(frac))
        mask = np.zeros((h, w), np.uint8)
        mask[h - band_h : h, :] = 255
        filled_bgr = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, inpaint_radius, cv2.INPAINT_TELEA)
        filled = filled_bgr[:, :, ::-1]  # BGR→RGB
        return filled, mask

    # ========== 背景補正/コントラスト ==========
    @staticmethod
    def flatfield(g: np.ndarray, sigma: int = 200) -> np.ndarray:
        """ガウシアンぼかしで背景を推定→引き算→正規化。背景ムラ除去。"""
        bg = cv2.GaussianBlur(g, (0, 0), sigma)
        x = cv2.subtract(g, bg)
        x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
        return x.astype(np.uint8)

    @staticmethod
    def rolling_ball_open(g: np.ndarray, radius: int = 101) -> np.ndarray:
        """形態学的Openで背景推定（ImageJのRolling Ball相当の簡易版）。"""
        k = max(3, int(radius) | 1)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, se)
        x = cv2.subtract(g, bg)
        x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
        return x.astype(np.uint8)

    @staticmethod
    def clahe(g: np.ndarray, clip: float = 2.0, tile: Tuple[int,int]=(16,16)) -> np.ndarray:
        """局所コントラスト強調（輝度むら影響の均し）。"""
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        return clahe.apply(g)

    # ========== バンドパス/エッジ強調 ==========
    @staticmethod
    def dog(g: np.ndarray, sigma_low: float = 2.0, sigma_high: float = 150.0) -> np.ndarray:
        """Difference of Gaussian（帯域抽出）。"""
        low  = cv2.GaussianBlur(g, (0,0), sigma_low)
        high = cv2.GaussianBlur(g, (0,0), sigma_high)
        band = cv2.subtract(low, high)
        band = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
        return band.astype(np.uint8)

    @staticmethod
    def gaussian_bandpass(g: np.ndarray, sigma_low: float = 2.0, sigma_high: float = 150.0) -> np.ndarray:
        """DoGと同等（名称だけ違うインターフェース）。"""
        return ImgPreproc.dog(g, sigma_low, sigma_high)

    # ========== ノイズ低減 ==========
    @staticmethod
    def nlm(x: np.ndarray, h: int = 10, template: int = 7, search: int = 21) -> np.ndarray:
        """Non-local Means（SEMのザラつきに有効）。"""
        return cv2.fastNlMeansDenoising(x, None, h=h, templateWindowSize=template, searchWindowSize=search)

    @staticmethod
    def median(x: np.ndarray, k: int = 3) -> np.ndarray:
        """塩胡椒ノイズに有効（副作用でエッジも丸まる）。"""
        return cv2.medianBlur(x, k)

    # ========== 二値化 ==========
    @staticmethod
    def otsu(x: np.ndarray) -> np.ndarray:
        """大津の二値化。"""
        _, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    @staticmethod
    def adaptive_gaussian(x: np.ndarray, block_size: int = 41, C: int = -2) -> np.ndarray:
        """適応的二値化（Gaussian）。block_sizeは奇数。"""
        return cv2.adaptiveThreshold(
            x, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, C
        )

    @staticmethod
    def seeded_threshold(x: np.ndarray, k_hi: float = 1.6, k_lo: float = 0.7) -> np.ndarray:
        """mean±kσを用いたシード付き二値化。"""
        m, s = float(x.mean()), float(x.std())
        t_hi = int(np.clip(m + k_hi*s, 1, 254))
        t_lo = int(np.clip(m + k_lo*s, 1, 254))
        _, strong = cv2.threshold(x, t_hi, 255, cv2.THRESH_BINARY)
        _, weak   = cv2.threshold(x, t_lo, 255, cv2.THRESH_BINARY)

        num, labels = cv2.connectedComponents(weak)
        keep = np.zeros_like(weak)
        if strong.any():
            strong_lbls = np.unique(labels[strong > 0])
            for l in strong_lbls:
                if l == 0: 
                    continue
                keep[labels == l] = 255
        return keep

    # ========== エッジ検出 ==========
    @staticmethod
    def canny(x: np.ndarray, t1: int = 40, t2: int = 100) -> np.ndarray:
        """Cannyエッジ。"""
        return cv2.Canny(x, t1, t2)

    # ========== 形態学 & マスク整形 ==========
    @staticmethod
    def morph_open(x: np.ndarray, k: int = 3) -> np.ndarray:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(x, cv2.MORPH_OPEN, se)

    @staticmethod
    def morph_close(x: np.ndarray, k: int = 5) -> np.ndarray:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(x, cv2.MORPH_CLOSE, se)

    @staticmethod
    def morph_and_fill(mask: np.ndarray, k_open: int = 3, k_close: int = 5) -> np.ndarray:
        """開閉→背景フラッドフィルで内穴を埋める。"""
        se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        se_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se_o)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se_c)
        h, w = m.shape
        ff = m.copy()
        pad = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(ff, pad, seedPoint=(0,0), newVal=255)
        holes = cv2.bitwise_not(ff)
        filled = cv2.bitwise_or(m, holes)
        return filled

    @staticmethod
    def filter_by_shape(mask: np.ndarray,
                        diam_range_px: Tuple[float,float]=(60,160),
                        circ_min: float = 0.60,
                        solidity_min: float = 0.85) -> np.ndarray:
        """面積（直径範囲換算）・円形度・充実度でフィルタ。"""
        dmin, dmax = diam_range_px
        amin = np.pi * (dmin/2)**2
        amax = np.pi * (dmax/2)**2
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < amin or area > amax:
                continue
            peri = cv2.arcLength(c, True) + 1e-6
            circ = 4*np.pi*area/(peri*peri)
            hull = cv2.convexHull(c); hull_area = cv2.contourArea(hull) + 1e-6
            solidity = area/hull_area
            if circ >= circ_min and solidity >= solidity_min:
                cv2.drawContours(out, [c], -1, 255, -1)
        return out

    @staticmethod
    def split_touching(mask: np.ndarray) -> np.ndarray:
        """距離変換＋ウォータシェッドで接触粒子を分離。"""
        if not mask.any():
            return mask
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_u8 = (dist / (dist.max() + 1e-6) * 255).astype(np.uint8)
        _, peaks = cv2.threshold(dist_u8, 90, 255, cv2.THRESH_BINARY)
        peaks = cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        _, markers = cv2.connectedComponents(peaks)
        markers = markers + 1
        markers[mask == 0] = 0
        ws = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
        out = np.where(ws > 1, 255, 0).astype(np.uint8)
        return out

    # ========== 可視化 ==========
    @staticmethod
    def overlay(rgb: np.ndarray, mask: np.ndarray,
                color: Tuple[int,int,int]=(0,255,0), thickness: int = 2) -> np.ndarray:
        """輪郭をRGB画像に描画。"""
        ov = rgb.copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, color, thickness)
        return ov
