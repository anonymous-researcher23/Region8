from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.readers import Scene, read_scene, inspect_raster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sanity-check one Sen1Floods11 HandLabeled scene: inspect rasters + read_scene() output."
    )
    p.add_argument(
        "--index_csv",
        type=str,
        default="data/processed/splits/sen1floods11_handlabeled_index.csv",
        help="Index CSV produced by build_index_sen1floods11.py",
    )
    p.add_argument(
        "--scene_id",
        type=str,
        required=True,
        help="Scene id to load (e.g., Ghana_24858). Paths + split are read from index_csv.",
    )
    return p.parse_args()


def load_scene_from_index(index_csv: Path, scene_id: str) -> Scene:
    index_csv = Path(index_csv).expanduser().resolve()
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    df = pd.read_csv(index_csv)
    required = {"scene_id", "s1_path", "s2_path", "label_path", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Index CSV missing columns: {sorted(missing)}")

    match = df[df["scene_id"] == scene_id]
    if match.shape[0] == 0:
        sample = df["scene_id"].head(15).tolist()
        raise ValueError(
            f"scene_id '{scene_id}' not found in {index_csv}. Example scene_ids: {sample}"
        )

    r = match.iloc[0]
    s1 = Path(r["s1_path"]).expanduser().resolve()
    s2 = Path(r["s2_path"]).expanduser().resolve()
    lb = Path(r["label_path"]).expanduser().resolve()
    split = str(r["split"])

    for pth in (s1, s2, lb):
        if not pth.exists():
            raise FileNotFoundError(f"Raster not found: {pth}")

    return Scene(scene_id=scene_id, s1_path=s1, s2_path=s2, label_path=lb, split=split)


def main():
    args = parse_args()
    scene_id = args.scene_id.strip()

    scene = load_scene_from_index(Path(args.index_csv), scene_id)

    print("\n=== Sanity Check: One Scene ===")
    print("scene_id:", scene.scene_id)
    print("split:", scene.split)
    print("S1:", scene.s1_path)
    print("S2:", scene.s2_path)
    print("Label:", scene.label_path)

    # Inspect rasters (band descriptions, nodata, etc.)
    print("\n--- Raster inspections ---")
    print("S1 inspect:", inspect_raster(scene.s1_path))
    print("S2 inspect:", inspect_raster(scene.s2_path))
    print("Label inspect:", inspect_raster(scene.label_path))

    # Read scene (returns X6, y, meta)
    X6, y, meta = read_scene(scene)

    print("\n--- read_scene outputs ---")
    print("X6 shape:", X6.shape, "(expected: (6, H, W))")
    print("y shape :", y.shape, "(expected: (H, W))")
    print("channel order:", meta.get("channel_order"))

    # Stats
    y_unique = np.unique(y)
    print("\ny unique (first up to 50):", y_unique[:50], "| count =", len(y_unique))

    # Quick min/max per channel for debugging (nan-safe)
    chans = meta.get("channel_order", ["VV", "VH", "B2", "B3", "B4", "B8"])
    print("\nX6 min/max per channel:")
    for i, ch in enumerate(chans):
        arr = X6[i]
        print(f"  {ch}: min={float(np.nanmin(arr)):.4f} max={float(np.nanmax(arr)):.4f}")

    print("\n✅ Sanity check complete.")


if __name__ == "__main__":
    main()
