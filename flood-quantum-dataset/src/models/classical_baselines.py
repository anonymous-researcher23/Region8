from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    log_loss,
)
import joblib


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train classical baselines (LogReg, RBF-SVM) on feature vectors in parquet.")
    ap.add_argument("--quantum_dir", type=str, default="data/processed/quantum/sen1floods11_handlabeled")
    ap.add_argument("--out_dir", type=str, default="outputs/models")

    ap.add_argument("--feature", type=str, default="z_hw",
                    help="Vector column to use (e.g., z_hw, z_sim, theta_hw, theta_region8, region8).")

    ap.add_argument("--svm_c", type=float, default=5.0)
    ap.add_argument("--svm_gamma", type=str, default="scale")

    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_val", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args()


def _load_split(path: Path, feature_col: str, max_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    if max_n and max_n > 0 and len(df) > max_n:
        df = df.sample(n=max_n, random_state=seed).reset_index(drop=True)

    if "y" not in df.columns:
        raise ValueError(f"'y' column missing in {path}")

    if feature_col not in df.columns:
        raise ValueError(f"feature='{feature_col}' not found in {path}. Columns: {df.columns.tolist()}")

    y = df["y"].to_numpy(dtype=np.int64)
    col = df[feature_col].values
    X = np.vstack([np.asarray(v, dtype=np.float32).ravel() for v in col]).astype(np.float32)
    return X, y


def _eval(name: str, y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> None:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    yhat = (p >= thr).astype(np.int64)

    acc = float(accuracy_score(y, yhat))
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    ll = float(log_loss(y, p, labels=[0, 1]))

    print(f"{name}: AUC={auc:.4f} logloss={ll:.4f} acc={acc:.4f} prec={pr:.4f} rec={rc:.4f} f1={f1:.4f}")


def main() -> None:
    args = parse_args()

    qdir = Path(args.quantum_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr_path = qdir / "train_quantum.parquet"
    va_path = qdir / "val_quantum.parquet"
    te_path = qdir / "test_quantum.parquet"

    for p in [tr_path, va_path, te_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print("Loading parquet splits...")
    Xtr, ytr = _load_split(tr_path, args.feature, args.max_train, args.seed)
    Xva, yva = _load_split(va_path, args.feature, args.max_val, args.seed)
    Xte, yte = _load_split(te_path, args.feature, args.max_test, args.seed)

    print(f"Using feature={args.feature} | shapes: train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    # Logistic Regression
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    logreg.fit(Xtr, ytr)
    pva_lr = logreg.predict_proba(Xva)[:, 1]
    pte_lr = logreg.predict_proba(Xte)[:, 1]

    print("\nLogReg")
    _eval("VAL", yva, pva_lr)
    _eval("TEST", yte, pte_lr)

    # RBF SVM
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=args.svm_c, gamma=args.svm_gamma, kernel="rbf", probability=True)),
    ])
    svm.fit(Xtr, ytr)
    pva_svm = svm.predict_proba(Xva)[:, 1]
    pte_svm = svm.predict_proba(Xte)[:, 1]

    print("\nRBF-SVM")
    _eval("VAL", yva, pva_svm)
    _eval("TEST", yte, pte_svm)

    # Save
    logreg_path = out / f"logreg_{args.feature}.joblib"
    svm_path = out / f"svm_rbf_{args.feature}.joblib"
    joblib.dump(logreg, logreg_path)
    joblib.dump(svm, svm_path)

    print(f"\nSaved:\n  {logreg_path}\n  {svm_path}")


if __name__ == "__main__":
    main()