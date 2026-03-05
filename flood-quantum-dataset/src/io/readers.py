from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import rasterio
from rasterio.warp import reproject, Resampling


# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class Scene:
    scene_id: str
    s1_path: Path
    s2_path: Path
    label_path: Path
    split: str  # train/val/test


# ----------------------------
# Index utilities
# ----------------------------
def load_index(index_csv: Path) -> List[Scene]:
    """
    Loads an index CSV with columns:
      scene_id, s1_path, s2_path, label_path, split
    """
    index_csv = Path(index_csv)
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    df = pd.read_csv(index_csv)
    required = {"scene_id", "s1_path", "s2_path", "label_path", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Index CSV missing columns: {sorted(missing)}")

    scenes: List[Scene] = []
    for _, r in df.iterrows():
        scenes.append(
            Scene(
                scene_id=str(r["scene_id"]),
                s1_path=Path(r["s1_path"]),
                s2_path=Path(r["s2_path"]),
                label_path=Path(r["label_path"]),
                split=str(r["split"]),
            )
        )
    return scenes


# ----------------------------
# Raster IO helpers
# ----------------------------
def _read_raster(path: Path) -> Tuple[np.ndarray, Dict[str, Any], Any, Any, Tuple[float, ...], Tuple[Optional[str], ...]]:
    """
    Returns:
      arr: (bands, H, W)
      profile: rasterio profile
      transform
      crs
      nodata per band (tuple)
      band_descriptions (tuple of strings or None)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as ds:
        arr = ds.read()  # (bands, H, W)
        profile = ds.profile.copy()
        transform = ds.transform
        crs = ds.crs
        nodata = ds.nodatavals  # tuple length = band count (can include None)
        band_desc = ds.descriptions  # tuple length = band count (strings or None)

    return arr, profile, transform, crs, nodata, band_desc


def _reproject_to_match(
    src_arr: np.ndarray,
    src_transform,
    src_crs,
    dst_shape_hw: Tuple[int, int],
    dst_transform,
    dst_crs,
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproject each band of src_arr to destination grid defined by (dst_shape_hw, dst_transform, dst_crs).
    src_arr shape: (bands, H, W)
    returns: (bands, dst_H, dst_W) float32
    """
    if src_arr.ndim != 3:
        raise ValueError(f"Expected src_arr (bands,H,W). Got shape {src_arr.shape}")

    bands, _, _ = src_arr.shape
    dst_h, dst_w = dst_shape_hw

    out = np.zeros((bands, dst_h, dst_w), dtype=np.float32)

    for b in range(bands):
        reproject(
            source=src_arr[b],
            destination=out[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return out


def inspect_raster(path: Path) -> Dict[str, Any]:
    """
    Quick helper to inspect band count and descriptions.
    Useful to confirm S1 band order (VV/VH).
    """
    arr, prof, transform, crs, nodata, desc = _read_raster(path)
    return {
        "path": str(path),
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "crs": str(crs),
        "nodata": nodata,
        "band_descriptions": desc,
        "transform": transform,
    }


# ----------------------------
# Main loader
# ----------------------------
def read_scene(scene: Scene) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Reads one Sen1Floods11 HandLabeled scene and returns:
      X6: (6, H, W) float32 in order [VV, VH, B2, B3, B4, B8]
      y : (H, W) uint8 flood mask (assumes label raster uses 1 for flood, 0 for non-flood)
      meta: dict with useful traceability info

    Notes:
      - Label grid defines target (H,W,transform,crs)
      - S1 and S2 are reprojected to label grid
      - S2 is expected to have 13 bands with descriptions including B1..B12; we extract B2,B3,B4,B8 by index.
    """
    # --- Read label first: target grid ---
    y_arr, y_prof, y_transform, y_crs, y_nodata, y_desc = _read_raster(scene.label_path)
    if y_arr.shape[0] < 1:
        raise ValueError(f"Label raster has no bands: {scene.label_path}")

    # Use first band as label
    y = y_arr[0].astype(np.uint8)

    # Target shape (H,W)
    H, W = y.shape

    # --- Read S1 ---
    s1_arr, _, s1_transform, s1_crs, s1_nodata, s1_desc = _read_raster(scene.s1_path)
    if s1_arr.shape[0] < 2:
        raise ValueError(
            f"S1 raster must have at least 2 bands (VV,VH). Got {s1_arr.shape[0]} bands: {scene.s1_path}"
        )

    # Reproject S1 to label grid (continuous values -> bilinear is fine)
    s1_on_y = _reproject_to_match(
        src_arr=s1_arr.astype(np.float32),
        src_transform=s1_transform,
        src_crs=s1_crs,
        dst_shape_hw=(H, W),
        dst_transform=y_transform,
        dst_crs=y_crs,
        resampling=Resampling.bilinear,
    )

    # --- Read S2 ---
    s2_arr, _, s2_transform, s2_crs, s2_nodata, s2_desc = _read_raster(scene.s2_path)
    if s2_arr.shape[0] < 8:
        raise ValueError(
            f"S2 raster must have >= 8 bands to extract B8. Got {s2_arr.shape[0]} bands: {scene.s2_path}"
        )

    # Reproject S2 to label grid (continuous values -> bilinear)
    s2_on_y = _reproject_to_match(
        src_arr=s2_arr.astype(np.float32),
        src_transform=s2_transform,
        src_crs=s2_crs,
        dst_shape_hw=(H, W),
        dst_transform=y_transform,
        dst_crs=y_crs,
        resampling=Resampling.bilinear,
    )

    # --- Select Sentinel-2 bands explicitly ---
    # Your S2Hand has 13 bands ordered: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
    # rasterio bands are read into numpy with 0-indexing:
    #   B1 -> 0, B2 -> 1, B3 -> 2, B4 -> 3, B8 -> 7
    B2 = s2_on_y[1]
    B3 = s2_on_y[2]
    B4 = s2_on_y[3]
    B8 = s2_on_y[7]
    optical = np.stack([B2, B3, B4, B8], axis=0).astype(np.float32)

    # --- Build final X6 in required order ---
    # NOTE: We assume S1 is [VV, VH] in the first two bands.
    # If your S1 is [VH, VV], swap them here once confirmed by inspecting band descriptions or docs.
    VV = s1_on_y[0]
    VH = s1_on_y[1]
    sar = np.stack([VV, VH], axis=0).astype(np.float32)

    X6 = np.concatenate([sar, optical], axis=0).astype(np.float32)

    # --- Basic sanity checks ---
    if X6.shape != (6, H, W):
        raise RuntimeError(f"Unexpected X6 shape {X6.shape}, expected (6,{H},{W}) for scene {scene.scene_id}")

    meta = {
        "scene_id": scene.scene_id,
        "split": scene.split,
        "paths": {
            "s1": str(scene.s1_path),
            "s2": str(scene.s2_path),
            "label": str(scene.label_path),
        },
        "target_grid": {
            "shape_hw": (H, W),
            "crs": str(y_crs),
            "transform": tuple(y_transform) if y_transform is not None else None,
            "label_nodata": y_nodata,
            "label_band_descriptions": y_desc,
        },
        "s1_info": {
            "band_descriptions": s1_desc,
            "nodata": s1_nodata,
            "bands_read": int(s1_arr.shape[0]),
        },
        "s2_info": {
            "band_descriptions": s2_desc,
            "nodata": s2_nodata,
            "bands_read": int(s2_arr.shape[0]),
            "selected": {"B2": 2, "B3": 3, "B4": 4, "B8": 8},  # 1-indexed band numbers for humans
        },
        "channel_order": ["VV", "VH", "B2", "B3", "B4", "B8"],
        "dtype": {"X6": str(X6.dtype), "y": str(y.dtype)},
    }

    return X6, y, meta


# ----------------------------
# Optional: convenience for single-file quick test
# ----------------------------
def quick_test_one_scene(scene_id: str, s1_path: str, s2_path: str, label_path: str) -> None:
    scene = Scene(scene_id=scene_id, s1_path=Path(s1_path), s2_path=Path(s2_path), label_path=Path(label_path), split="na")
    X6, y, meta = read_scene(scene)
    print("X6:", X6.shape, X6.dtype, "min/max:", float(np.nanmin(X6)), float(np.nanmax(X6)))
    print("y :", y.shape, y.dtype, "unique:", np.unique(y)[:20])
    print("channel order:", meta["channel_order"])
    print("S2 band desc:", meta["s2_info"]["band_descriptions"])
    print("S1 band desc:", meta["s1_info"]["band_descriptions"])
