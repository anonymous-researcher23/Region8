# scripts/build_region8_features.py
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.preprocess.region_features import (
    BandOrder,
    load_patch,
    compute_region8_features,
    fit_standardizer,
    apply_standardizer,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with train.parquet/val.parquet/test.parquet (patch index)")
    ap.add_argument("--out_dir", required=True, help="Where to write region8 feature parquets + stats json")
    ap.add_argument("--x_col", default="x_path", help="Column containing path to patch array")
    ap.add_argument("--region_grid", default="4,2", help="Grid for pooling, e.g. '4,2' gives 8 regions")
    ap.add_argument("--band_order", default="0,1,2,3,4,5", help="vv,vh,b2,b3,b4,b8 indices")
    ap.add_argument("--write_z", action="store_true", help="Also write standardized region8 as region8_z")
    ap.add_argument("--max_sar_nan_frac", type=float, default=0.50, help="Skip patch if max(VV_nan,VH_nan) exceeds this")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = tuple(int(x) for x in args.region_grid.split(","))
    bo = tuple(int(x) for x in args.band_order.split(","))
    if len(bo) != 6:
        raise ValueError("--band_order must have 6 ints: vv,vh,b2,b3,b4,b8")
    band_order = BandOrder(vv=bo[0], vh=bo[1], b2=bo[2], b3=bo[3], b4=bo[4], b8=bo[5])

    # Load split parquets
    splits: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        p = in_dir / f"{split}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        df = pd.read_parquet(p)
        if args.x_col not in df.columns:
            raise ValueError(f"Missing column '{args.x_col}' in {p}. Columns={list(df.columns)}")
        splits[split] = df

    def extract(df: pd.DataFrame, split_name: str):
        feats = []
        keep_idx = []
        bad = []

        xpaths = df[args.x_col].astype(str).tolist()
        for i, xp in enumerate(xpaths):
            try:
                x = load_patch(xp)
                f = compute_region8_features(
                    x,
                    band_order=band_order,
                    region_grid=grid,
                    max_sar_nan_frac=args.max_sar_nan_frac,
                )
                if not np.isfinite(f).all():
                    bad.append((split_name, i, xp, "nonfinite_features"))
                    continue
                feats.append(f)
                keep_idx.append(i)
            except Exception as e:
                bad.append((split_name, i, xp, f"{type(e).__name__}:{e}"))
                continue

        X = np.asarray(feats, dtype=np.float32)
        df_keep = df.iloc[keep_idx].reset_index(drop=True)
        return df_keep, X, bad

    # Extract features, filtering bad patches
    df_train, X_train, bad_train = extract(splits["train"], "train")
    df_val,   X_val,   bad_val   = extract(splits["val"], "val")
    df_test,  X_test,  bad_test  = extract(splits["test"], "test")
    bad_all = bad_train + bad_val + bad_test

    # Log skipped patches
    if bad_all:
        bad_df = pd.DataFrame(bad_all, columns=["split", "row_index", "x_path", "reason"])
        bad_csv = out_dir / "bad_patches_region8.csv"
        bad_df.to_csv(bad_csv, index=False)
        print(f"Skipped {len(bad_df)} bad patches. Logged to {bad_csv}")

    if len(df_train) == 0:
        raise RuntimeError("All train patches were filtered out. Lower strictness or inspect patch generation.")

    # Save raw region8 as list column
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["region8"] = [row.tolist() for row in X_train]
    df_val["region8"] = [row.tolist() for row in X_val]
    df_test["region8"] = [row.tolist() for row in X_test]

    # Fit standardizer on TRAIN ONLY
    stats = fit_standardizer(X_train)

    # Save stats
    stats_path = out_dir / "region8_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "mean": stats["mean"].tolist(),
                "std": stats["std"].tolist(),
                "region_grid": list(grid),
                "band_order": list(bo),
                "max_sar_nan_frac": args.max_sar_nan_frac,
            },
            f,
            indent=2,
        )

    # Optionally add standardized column
    if args.write_z:
        X_train_z = apply_standardizer(X_train, stats)
        X_val_z = apply_standardizer(X_val, stats)
        X_test_z = apply_standardizer(X_test, stats)

        df_train["region8_z"] = [row.tolist() for row in X_train_z]
        df_val["region8_z"] = [row.tolist() for row in X_val_z]
        df_test["region8_z"] = [row.tolist() for row in X_test_z]

    # Write output parquets
    (out_dir / "train.parquet").parent.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(out_dir / "train.parquet", index=False)
    df_val.to_parquet(out_dir / "val.parquet", index=False)
    df_test.to_parquet(out_dir / "test.parquet", index=False)

    print("Wrote:")
    print(" ", out_dir / "train.parquet")
    print(" ", out_dir / "val.parquet")
    print(" ", out_dir / "test.parquet")
    print(" ", stats_path)


if __name__ == "__main__":
    main()