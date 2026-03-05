from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


EXPECTED_SHAPE = (6, 32, 32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Flatten normalized patches, fit PCA on train, transform all splits, standardize train-only.")
    ap.add_argument(
        "--norm_dir",
        type=str,
        default="data/processed/normalized/sen1floods11_handlabeled",
        help="Directory containing normalized train/val/test parquet with x_path pointing to normalized .npy patches",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/features/sen1floods11_handlabeled",
        help="Output directory for PCA features and metadata",
    )
    ap.add_argument(
        "--dims_sim",
        type=int,
        default=16,
        help="PCA dimensionality for simulation",
    )
    ap.add_argument(
        "--dims_hw",
        type=int,
        default=8,
        help="PCA dimensionality for hardware",
    )
    ap.add_argument(
        "--x_col",
        type=str,
        default="x_path",
        help="Column in parquet that points to .npy patch files",
    )
    ap.add_argument(
        "--y_col",
        type=str,
        default="y",
        help="Column in parquet for label (0/1)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=4096,
        help="Batch size for loading patches",
    )
    return ap.parse_args()


def load_patch(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    if x.shape != EXPECTED_SHAPE:
        raise ValueError(f"Bad patch shape {x.shape} for {path}, expected {EXPECTED_SHAPE}")
    # Replace NaNs with 0.0 AFTER normalization (safe default for PCA)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def flatten_patch(x: np.ndarray) -> np.ndarray:
    # (6,32,32) -> (6144,)
    return x.reshape(-1)


def load_flat_matrix(df: pd.DataFrame, x_col: str, batch: int) -> np.ndarray:
    N = len(df)
    X = np.empty((N, 6 * 32 * 32), dtype=np.float32)

    for start in range(0, N, batch):
        end = min(N, start + batch)
        paths = df.iloc[start:end][x_col].tolist()
        for i, p in enumerate(paths, start=start):
            x = load_patch(Path(p))
            X[i] = flatten_patch(x)

        print(f"  loaded {end}/{N}")

    return X


def fit_pca(X_train: np.ndarray, n_components: int) -> PCA:
    # randomized solver is faster for big matrices
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1337)
    pca.fit(X_train)
    return pca


def standardize_train_only(Z_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = Z_train.mean(axis=0)
    sigma = Z_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Zs = (Z_train - mu) / sigma
    return Zs, mu, sigma


def apply_standardize(Z: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (Z - mu) / sigma


def save_split_features(
    df: pd.DataFrame,
    Z_sim: np.ndarray,
    Z_hw: np.ndarray,
    out_path: Path,
    y_col: str,
) -> None:
    out = pd.DataFrame({
        "scene_id": df["scene_id"].astype(str),
        "patch_id": df["patch_id"].astype(str),
        "split": df["split"].astype(str),
        "row": df["row"].astype(int),
        "col": df["col"].astype(int),
        "flood_frac": df["flood_frac"].astype(float),
        "y": df[y_col].astype(int),
        "z_sim": list(Z_sim.astype(np.float32)),
        "z_hw": list(Z_hw.astype(np.float32)),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"✅ wrote {out_path} | N={len(out)}")


def main():
    args = parse_args()

    norm_dir = Path(args.norm_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    train_pq = norm_dir / "train.parquet"
    val_pq = norm_dir / "val.parquet"
    test_pq = norm_dir / "test.parquet"
    for p in (train_pq, val_pq, test_pq):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print("Reading normalized parquet tables...")
    df_train = pd.read_parquet(train_pq)
    df_val = pd.read_parquet(val_pq)
    df_test = pd.read_parquet(test_pq)

    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for col in [args.x_col, args.y_col, "scene_id", "patch_id", "split", "row", "col", "flood_frac"]:
            if col not in df.columns:
                raise ValueError(f"{name}.parquet missing '{col}'. Columns: {list(df.columns)}")

    print(f"Train N={len(df_train)} | Val N={len(df_val)} | Test N={len(df_test)}")

    print("\nLoading flattened matrices (6144-D)...")
    print("TRAIN:")
    X_train = load_flat_matrix(df_train, args.x_col, args.batch)
    print("VAL:")
    X_val = load_flat_matrix(df_val, args.x_col, args.batch)
    print("TEST:")
    X_test = load_flat_matrix(df_test, args.x_col, args.batch)

    print("\nFitting PCA on TRAIN only...")
    pca_sim = fit_pca(X_train, args.dims_sim)
    pca_hw = fit_pca(X_train, args.dims_hw)

    print("Transforming splits...")
    Z_train_sim = pca_sim.transform(X_train).astype(np.float32)
    Z_val_sim = pca_sim.transform(X_val).astype(np.float32)
    Z_test_sim = pca_sim.transform(X_test).astype(np.float32)

    Z_train_hw = pca_hw.transform(X_train).astype(np.float32)
    Z_val_hw = pca_hw.transform(X_val).astype(np.float32)
    Z_test_hw = pca_hw.transform(X_test).astype(np.float32)

    print("\nStandardizing using TRAIN mean/std only...")
    Z_train_sim_s, mu_sim, sig_sim = standardize_train_only(Z_train_sim)
    Z_val_sim_s = apply_standardize(Z_val_sim, mu_sim, sig_sim)
    Z_test_sim_s = apply_standardize(Z_test_sim, mu_sim, sig_sim)

    Z_train_hw_s, mu_hw, sig_hw = standardize_train_only(Z_train_hw)
    Z_val_hw_s = apply_standardize(Z_val_hw, mu_hw, sig_hw)
    Z_test_hw_s = apply_standardize(Z_test_hw, mu_hw, sig_hw)

    # Save PCA artifacts + stats
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save PCA components + explained variance (portable)
    pca_pack = {
        "dims_sim": args.dims_sim,
        "dims_hw": args.dims_hw,
        "pca_sim_components": pca_sim.components_.tolist(),
        "pca_sim_mean": pca_sim.mean_.tolist(),
        "pca_sim_explained_variance_ratio": pca_sim.explained_variance_ratio_.tolist(),
        "pca_hw_components": pca_hw.components_.tolist(),
        "pca_hw_mean": pca_hw.mean_.tolist(),
        "pca_hw_explained_variance_ratio": pca_hw.explained_variance_ratio_.tolist(),
        "standardize_sim": {"mean": mu_sim.tolist(), "std": sig_sim.tolist()},
        "standardize_hw": {"mean": mu_hw.tolist(), "std": sig_hw.tolist()},
        "note": "PCA fit on TRAIN only. Standardization uses TRAIN mean/std only.",
    }
    (out_dir / "pca_and_standardize.json").write_text(json.dumps(pca_pack, indent=2))
    print(f"\n✅ wrote {(out_dir / 'pca_and_standardize.json')}")

    # Save split feature tables
    save_split_features(df_train, Z_train_sim_s, Z_train_hw_s, out_dir / "train_features.parquet", args.y_col)
    save_split_features(df_val, Z_val_sim_s, Z_val_hw_s, out_dir / "val_features.parquet", args.y_col)
    save_split_features(df_test, Z_test_sim_s, Z_test_hw_s, out_dir / "test_features.parquet", args.y_col)

    # Quick diagnostic
    print("\nDiagnostics:")
    print(f"  sim explained var (sum): {float(np.sum(pca_sim.explained_variance_ratio_)):.4f}")
    print(f"  hw  explained var (sum): {float(np.sum(pca_hw.explained_variance_ratio_)):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
