from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.io.readers import load_index, read_scene

PATCH = 32
POS_THR = 0.10
NEG_THR = 0.01

# If labels are not exactly 0/1, map them here:
# Example: flood=255 => map to 1
def normalize_label(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.int32)
    # Common cases: {0,1} or {0,255}
    if set(np.unique(y)).issubset({0, 1}):
        return y.astype(np.uint8)
    if set(np.unique(y)).issubset({0, 255}):
        return (y == 255).astype(np.uint8)
    # If multi-class: treat "flood" as non-zero (conservative)
    return (y > 0).astype(np.uint8)

def iter_patches(X6: np.ndarray, y: np.ndarray, scene_id: str, split: str):
    _, H, W = X6.shape
    # Non-overlapping tiling
    for r in range(0, H - PATCH + 1, PATCH):
        for c in range(0, W - PATCH + 1, PATCH):
            x_patch = X6[:, r:r+PATCH, c:c+PATCH]
            y_patch = y[r:r+PATCH, c:c+PATCH]

            # Skip empty/invalid patches (optional)
            # Example: if lots of zeros are nodata, you can add a rule here.
            flood_frac = float(y_patch.mean())
            if flood_frac >= POS_THR:
                label = 1
            elif flood_frac <= NEG_THR:
                label = 0
            else:
                continue

            patch_id = f"{scene_id}_{r}_{c}"
            yield {
                "scene_id": scene_id,
                "patch_id": patch_id,
                "row": r,
                "col": c,
                "split": split,
                "flood_frac": flood_frac,
                "y": label,
                # store patch array in memory for later saving
                "x_patch": x_patch.astype(np.float32),
            }

def main():
    index_csv = Path("data/processed/splits/sen1floods11_handlabeled_index.csv")
    out_dir = Path("data/processed/patches/sen1floods11_handlabeled")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_index(index_csv)

    # We’ll write per split
    buffers = {"train": [], "val": [], "test": []}

    for scene in scenes:
        X6, y_raw, meta = read_scene(scene)
        y = normalize_label(y_raw)

        # Generate patches
        for rec in iter_patches(X6, y, scene.scene_id, scene.split):
            buffers[scene.split].append(rec)

        print(f"Done {scene.scene_id} ({scene.split}) | patches so far:",
              {k: len(v) for k,v in buffers.items()})

    # Save patches: store X as separate .npy files and save metadata table
    # This avoids huge parquet files with arrays inside.
    for split, recs in buffers.items():
        if not recs:
            continue

        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for rec in recs:
            x = rec.pop("x_patch")
            x_path = split_dir / f"{rec['patch_id']}.npy"
            np.save(x_path, x)
            rec["x_path"] = str(x_path)
            rows.append(rec)

        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / f"{split}.parquet", index=False)
        print(f"Saved {split}: {len(df)} patches -> {out_dir / f'{split}.parquet'}")

if __name__ == "__main__":
    main()
