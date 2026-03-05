from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Encode PCA features into angle + amplitude representations for quantum models.")
    ap.add_argument(
        "--feat_dir",
        type=str,
        default="data/processed/features/sen1floods11_handlabeled",
        help="Directory containing *_features.parquet produced by build_features_pca.py",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/quantum/sen1floods11_handlabeled",
        help="Output directory for quantum-encoded parquet shards.",
    )
    ap.add_argument(
        "--clip",
        type=float,
        default=3.0,
        help="Clip value for standardized features before mapping to angles.",
    )
    ap.add_argument(
        "--angle_max",
        type=float,
        default=math.pi / 2.0,
        help="Maximum absolute angle (radians). Default pi/2.",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small epsilon to avoid divide-by-zero in amplitude normalization.",
    )
    return ap.parse_args()


def to_numpy_stack(series) -> np.ndarray:
    # parquet stores list/array objects; convert to stacked float32 matrix
    return np.stack(series.to_list(), axis=0).astype(np.float32)


def angle_encode(Z: np.ndarray, clip: float, angle_max: float) -> np.ndarray:
    # z -> theta in [-angle_max, angle_max]
    Zc = np.clip(Z, -clip, clip)
    theta = Zc * (angle_max / clip)
    return theta.astype(np.float32)


def amplitude_encode(V: np.ndarray, eps: float) -> np.ndarray:
    """
    V: (N, 8) -> normalized amplitudes (N, 8) with L2 norm = 1.
    For 8 dims -> 3 qubits (2^3).
    """
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    A = V / norms
    return A.astype(np.float32)


def process_split(split_name: str, in_path: Path, out_path: Path, clip: float, angle_max: float, eps: float) -> None:
    df = pd.read_parquet(in_path)

    required = ["scene_id", "patch_id", "split", "row", "col", "flood_frac", "y", "z_sim", "z_hw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{in_path} missing columns: {missing}")

    Z_sim = to_numpy_stack(df["z_sim"])  # (N,16)
    Z_hw = to_numpy_stack(df["z_hw"])    # (N,8)

    theta_sim = angle_encode(Z_sim, clip=clip, angle_max=angle_max)  # (N,16)
    theta_hw = angle_encode(Z_hw, clip=clip, angle_max=angle_max)    # (N,8)

    amp_hw = amplitude_encode(Z_hw, eps=eps)  # (N,8)

    out = pd.DataFrame({
        "scene_id": df["scene_id"].astype(str),
        "patch_id": df["patch_id"].astype(str),
        "split": df["split"].astype(str),
        "row": df["row"].astype(int),
        "col": df["col"].astype(int),
        "flood_frac": df["flood_frac"].astype(float),
        "y": df["y"].astype(int),

        # Classical / traceability
        "z_sim": list(Z_sim),
        "z_hw": list(Z_hw),

        # Quantum encodings
        "theta_sim": list(theta_sim),  # for 16-qubit sim
        "theta_hw": list(theta_hw),    # for 8-qubit hw
        "amp_hw": list(amp_hw),        # for 3-qubit amplitude encoding demo
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    # quick checks
    a = np.stack(out["amp_hw"].to_list(), axis=0)
    norms = np.linalg.norm(a, axis=1)
    print(f"✅ {split_name}: wrote {out_path} | N={len(out)} | amp norms: min={norms.min():.6f} max={norms.max():.6f}")


def main():
    args = parse_args()

    feat_dir = Path(args.feat_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    in_train = feat_dir / "train_features.parquet"
    in_val = feat_dir / "val_features.parquet"
    in_test = feat_dir / "test_features.parquet"
    for p in (in_train, in_val, in_test):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    out_train = out_dir / "train_quantum.parquet"
    out_val = out_dir / "val_quantum.parquet"
    out_test = out_dir / "test_quantum.parquet"

    print("Encoding quantum representations...")
    process_split("train", in_train, out_train, args.clip, args.angle_max, args.eps)
    process_split("val", in_val, out_val, args.clip, args.angle_max, args.eps)
    process_split("test", in_test, out_test, args.clip, args.angle_max, args.eps)

    print("\nDone.")


if __name__ == "__main__":
    main()
