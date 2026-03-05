from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


CHANNELS = ["VV", "VH", "B2", "B3", "B4", "B8"]
EXPECTED_SHAPE = (6, 32, 32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute train-only p1/p99 stats and normalize patches saved as .npy referenced by x_path in parquet."
    )
    ap.add_argument(
        "--patch_dir",
        type=str,
        default="data/processed/patches/sen1floods11_handlabeled",
        help="Directory containing train.parquet/val.parquet/test.parquet and per-split .npy patch files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/normalized/sen1floods11_handlabeled",
        help="Output root for normalized patches and updated parquet files.",
    )
    ap.add_argument(
        "--stats_out",
        type=str,
        default="data/processed/stats/sen1floods11_handlabeled_p1p99.json",
        help="Where to write JSON stats (computed on TRAIN only).",
    )
    ap.add_argument(
        "--x_col",
        type=str,
        default="x_path",
        help="Column name in parquet that points to .npy patch file.",
    )
    ap.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="Optional nodata sentinel value to ignore in stats (besides NaN/Inf).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrites existing normalized .npy files.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="If set, runs quick sanity checks after normalization.",
    )
    return ap.parse_args()


def load_patch_npy(path: Path) -> np.ndarray:
    x = np.load(path)
    x = np.asarray(x, dtype=np.float32)
    if x.shape != EXPECTED_SHAPE:
        raise ValueError(f"Patch shape mismatch for {path}: got {x.shape}, expected {EXPECTED_SHAPE}")
    return x


def valid_values(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    v = arr.reshape(-1)
    mask = np.isfinite(v)
    if nodata is not None:
        mask = mask & (v != nodata)
    v = v[mask]
    return v


def compute_train_p1p99(train_df: pd.DataFrame, x_col: str, nodata: Optional[float]) -> Dict[str, Dict[str, float]]:
    # Collect per-channel values (list of arrays), then concat.
    # This is fine for many patch datasets; if yours is massive, we can switch to streaming quantiles.
    per_ch: List[List[np.ndarray]] = [[] for _ in range(6)]

    total = len(train_df)
    if total == 0:
        raise RuntimeError("Train parquet is empty — no patches to compute stats on.")

    for i, row in enumerate(train_df.itertuples(index=False), start=1):
        xp = Path(getattr(row, x_col))
        x = load_patch_npy(xp)

        for ci in range(6):
            v = valid_values(x[ci], nodata)
            if v.size > 0:
                per_ch[ci].append(v)

        if i % 2000 == 0:
            print(f"  stats pass: {i}/{total} patches")

    stats: Dict[str, Dict[str, float]] = {}
    for ci, ch in enumerate(CHANNELS):
        if len(per_ch[ci]) == 0:
            raise RuntimeError(f"No valid pixels found for channel {ch}. Check nodata handling.")
        all_v = np.concatenate(per_ch[ci], axis=0)
        p1 = float(np.percentile(all_v, 1.0))
        p99 = float(np.percentile(all_v, 99.0))
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            raise RuntimeError(f"Bad percentiles for {ch}: p1={p1}, p99={p99}")
        # avoid degenerate
        if np.isclose(p1, p99):
            p99 = p1 + 1e-6
        stats[ch] = {"p1": p1, "p99": p99}

    return stats


def normalize_patch(x: np.ndarray, stats: Dict[str, Dict[str, float]], nodata: Optional[float]) -> np.ndarray:
    out = x.astype(np.float32, copy=True)

    for ci, ch in enumerate(CHANNELS):
        p1 = stats[ch]["p1"]
        p99 = stats[ch]["p99"]

        xb = out[ci]
        invalid = ~np.isfinite(xb)
        if nodata is not None:
            invalid = invalid | (xb == nodata)

        xb = np.clip(xb, p1, p99)
        xb = (xb - p1) / (p99 - p1)   # [0,1]
        xb = xb * 2.0 - 1.0           # [-1,1]
        xb = xb.astype(np.float32)

        xb[invalid] = np.nan
        out[ci] = xb

    return out


def process_split(
    split_name: str,
    df: pd.DataFrame,
    x_col: str,
    stats: Dict[str, Dict[str, float]],
    nodata: Optional[float],
    out_root: Path,
    overwrite: bool,
) -> pd.DataFrame:
    """
    Writes normalized .npy patches under:
      out_root / split_name / <patch_id>.npy
    and returns updated dataframe with x_path pointing to normalized files.
    """
    out_split_dir = out_root / split_name
    out_split_dir.mkdir(parents=True, exist_ok=True)

    updated = df.copy()

    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        in_path = Path(getattr(row, x_col))
        if not in_path.exists():
            raise FileNotFoundError(f"Missing patch file: {in_path}")

        # Choose output name:
        # Prefer patch_id if present, else use original filename.
        patch_id = getattr(row, "patch_id", None)
        if patch_id is None or str(patch_id).strip() == "":
            out_name = in_path.name
        else:
            out_name = f"{patch_id}.npy"

        out_path = out_split_dir / out_name

        if out_path.exists() and not overwrite:
            # Just point to it
            updated.loc[i - 1, x_col] = str(out_path)
            continue

        x = load_patch_npy(in_path)
        xn = normalize_patch(x, stats, nodata)
        np.save(out_path, xn)

        updated.loc[i - 1, x_col] = str(out_path)

        if i % 2000 == 0:
            print(f"  {split_name}: {i}/{total} normalized")

    return updated


def sanity_check_random(df: pd.DataFrame, x_col: str, n: int = 10) -> None:
    if len(df) == 0:
        print("Sanity check skipped: empty split.")
        return
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=1337)

    mins = []
    maxs = []
    for row in sample.itertuples(index=False):
        x = np.load(Path(getattr(row, x_col))).astype(np.float32)
        mins.append(float(np.nanmin(x)))
        maxs.append(float(np.nanmax(x)))

    print(f"Sanity check ({n} patches): min range [{min(mins):.4f}, {max(mins):.4f}] "
          f"max range [{min(maxs):.4f}, {max(maxs):.4f}]")
    if min(mins) < -1.05 or max(maxs) > 1.05:
        raise RuntimeError("Normalized values out of expected [-1,1] bounds (tolerance 0.05).")


def main():
    args = parse_args()

    patch_dir = Path(args.patch_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    stats_out = Path(args.stats_out).expanduser().resolve()

    train_pq = patch_dir / "train.parquet"
    val_pq = patch_dir / "val.parquet"
    test_pq = patch_dir / "test.parquet"

    for p in (train_pq, val_pq, test_pq):
        if not p.exists():
            raise FileNotFoundError(f"Missing parquet file: {p}")

    print("Reading parquet tables...")
    df_train = pd.read_parquet(train_pq)
    df_val = pd.read_parquet(val_pq)
    df_test = pd.read_parquet(test_pq)

    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if args.x_col not in df.columns:
            raise ValueError(f"{name}.parquet missing required column '{args.x_col}'. Columns: {list(df.columns)}")

    print(f"Train patches: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    print("\nComputing TRAIN-only p1/p99 stats...")
    stats = compute_train_p1p99(df_train, x_col=args.x_col, nodata=args.nodata)

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "channels": CHANNELS,
        "stats": stats,
        "fit_on": "train",
        "target_range": [-1.0, 1.0],
        "note": "Clip to [p1,p99] then scale to [-1,1] per channel. NaNs preserved.",
    }
    stats_out.write_text(json.dumps(payload, indent=2))
    print(f"\n✅ Wrote stats: {stats_out}")
    for ch in CHANNELS:
        print(f"  {ch}: p1={stats[ch]['p1']:.6f}  p99={stats[ch]['p99']:.6f}")

    print("\nNormalizing splits and writing normalized .npy files...")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_n = process_split("train", df_train, args.x_col, stats, args.nodata, out_dir, args.overwrite)
    df_val_n = process_split("val", df_val, args.x_col, stats, args.nodata, out_dir, args.overwrite)
    df_test_n = process_split("test", df_test, args.x_col, stats, args.nodata, out_dir, args.overwrite)

    # Write updated parquet tables next to normalized output
    train_out_pq = out_dir / "train.parquet"
    val_out_pq = out_dir / "val.parquet"
    test_out_pq = out_dir / "test.parquet"

    df_train_n.to_parquet(train_out_pq, index=False)
    df_val_n.to_parquet(val_out_pq, index=False)
    df_test_n.to_parquet(test_out_pq, index=False)

    print("\n✅ Wrote updated parquet tables:")
    print(f"  {train_out_pq}")
    print(f"  {val_out_pq}")
    print(f"  {test_out_pq}")

    if args.verify:
        print("\nRunning sanity checks on normalized outputs...")
        sanity_check_random(df_train_n, args.x_col, n=10)
        sanity_check_random(df_val_n, args.x_col, n=10)
        sanity_check_random(df_test_n, args.x_col, n=10)
        print("✅ Sanity checks passed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
