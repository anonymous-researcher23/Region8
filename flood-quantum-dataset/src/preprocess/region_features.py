# src/preprocess/region_features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Dict

import numpy as np


@dataclass(frozen=True)
class BandOrder:
    """
    Indices for each band in your patch tensor.
    Assumes patch is either (C,H,W) or (H,W,C).
    Default order: [VV, VH, B2, B3, B4, B8]
    """
    vv: int = 0
    vh: int = 1
    b2: int = 2
    b3: int = 3
    b4: int = 4
    b8: int = 5


def _ensure_chw(x: np.ndarray) -> np.ndarray:
    """
    Accept (C,H,W) or (H,W,C) and return (C,H,W).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={x.shape}")

    # If first dim looks like channels
    if x.shape[0] <= 16 and x.shape[1] > 16 and x.shape[2] > 16:
        return x

    # If last dim looks like channels
    if x.shape[2] <= 16 and x.shape[0] > 16 and x.shape[1] > 16:
        return np.transpose(x, (2, 0, 1))

    raise ValueError(f"Cannot infer channel order from shape={x.shape}")


def load_patch(x_path: str | Path) -> np.ndarray:
    """
    Load a single patch from disk.
    Supports .npy and .npz (expects array under key 'x' or first array).
    """
    p = Path(x_path)
    if not p.exists():
        raise FileNotFoundError(f"Patch file not found: {p}")

    if p.suffix == ".npy":
        return np.load(p)

    if p.suffix == ".npz":
        z = np.load(p)
        if "x" in z:
            return z["x"]
        for k in z.files:
            return z[k]
        raise ValueError(f"Empty npz: {p}")

    raise ValueError(f"Unsupported patch format: {p.suffix} ({p})")


def split_regions(H: int, W: int, grid: Tuple[int, int]) -> Iterable[Tuple[slice, slice]]:
    """
    Yield region slices for a grid=(gh,gw). For (4,2) you get 8 regions.
    Requires H divisible by gh and W divisible by gw (32 works).
    """
    gh, gw = grid
    if H % gh != 0 or W % gw != 0:
        raise ValueError(f"H,W must be divisible by grid. H={H}, W={W}, grid={grid}")

    rh = H // gh
    rw = W // gw

    for i in range(gh):
        for j in range(gw):
            rs = slice(i * rh, (i + 1) * rh)
            cs = slice(j * rw, (j + 1) * rw)
            yield rs, cs


def sar_nan_fraction(x_chw: np.ndarray, vv_idx: int, vh_idx: int) -> float:
    """
    Max fraction of NaNs across VV/VH channels.
    """
    vv = x_chw[vv_idx]
    vh = x_chw[vh_idx]
    return float(max(np.isnan(vv).mean(), np.isnan(vh).mean()))


def compute_region8_features(
    x: np.ndarray,
    band_order: BandOrder = BandOrder(),
    region_grid: Tuple[int, int] = (4, 2),
    eps: float = 1e-6,
    max_sar_nan_frac: float = 0.50,
) -> np.ndarray:
    """
    Compute 8 spatial region features from one patch.

    Strategy:
      - Keep patches with small SAR NaN holes (replace NaNs with 0)
      - Drop patches where SAR is mostly missing (raise ValueError)
      - Features per region: mean(NDWI_proxy) + 0.25 * mean(log_ratio_VV_VH)

    Returns: (8,) float32
    """
    x = _ensure_chw(x).astype(np.float32, copy=False)
    C, H, W = x.shape

    # Reject patches where SAR is largely missing (e.g., VV/VH all NaN)
    frac = sar_nan_fraction(x, band_order.vv, band_order.vh)
    if frac > max_sar_nan_frac:
        raise ValueError(f"Invalid SAR: nan_frac={frac:.3f} > {max_sar_nan_frac}")

    # Sanitize partial NaNs/Infs
    vv = np.nan_to_num(x[band_order.vv], nan=0.0, posinf=0.0, neginf=0.0)
    vh = np.nan_to_num(x[band_order.vh], nan=0.0, posinf=0.0, neginf=0.0)
    b3 = np.nan_to_num(x[band_order.b3], nan=0.0, posinf=0.0, neginf=0.0)
    b8 = np.nan_to_num(x[band_order.b8], nan=0.0, posinf=0.0, neginf=0.0)

    # NDWI proxy (Green - NIR)/(Green + NIR + eps) but stabilized
    denom = b3 + b8
    denom = np.where(np.abs(denom) < eps, eps, denom)
    ndwi = (b3 - b8) / denom
    ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)

    # SAR log ratio proxy
    log_ratio = np.log(np.abs(vv) + eps) - np.log(np.abs(vh) + eps)
    log_ratio = np.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)

    feats = []
    for rs, cs in split_regions(H, W, region_grid):
        ndwi_mean = float(np.mean(ndwi[rs, cs]))
        lr_mean = float(np.mean(log_ratio[rs, cs]))
        feats.append(ndwi_mean + 0.25 * lr_mean)

    out = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fit_standardizer(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    X: (N,8)
    returns dict with mean/std (train-only)
    """
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def apply_standardizer(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((X - stats["mean"]) / stats["std"]).astype(np.float32)