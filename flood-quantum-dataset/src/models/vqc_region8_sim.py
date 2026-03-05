from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli


# -------------------------
# Utils
# -------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def set_seed(seed: int) -> None:
    # NumPy
    np.random.seed(seed)
    # Python random (just in case you add anything later that uses it)
    random.seed(seed)
    # Optional: hash seed for deterministic hashing behavior
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_labels(df: pd.DataFrame, label_col: str = "y") -> np.ndarray:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in parquet.")
    return df[label_col].to_numpy(dtype=np.int64)


def get_feature_matrix(df: pd.DataFrame, feature_col: str) -> np.ndarray:
    if feature_col not in df.columns:
        raise ValueError(
            f"feature_col='{feature_col}' not found. Available columns: {df.columns.tolist()}"
        )
    col = df[feature_col].values
    X = np.vstack([np.asarray(v, dtype=np.float32).ravel() for v in col]).astype(np.float32)
    return X


def standardize_train_only(
    Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def best_threshold_by_f1(y_true: np.ndarray, p: np.ndarray, steps: int = 401) -> Tuple[float, float]:
    best_thr = 0.5
    best = -1.0
    for t in np.linspace(0.0, 1.0, steps):
        yhat = (p >= t).astype(np.int64)
        f1 = f1_score(y_true, yhat, zero_division=0)
        if f1 > best:
            best = f1
            best_thr = float(t)
    return best_thr, float(best)


def eval_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    yhat = (p >= thr).astype(np.int64)

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    ll = float(log_loss(y, p, labels=[0, 1]))
    acc = float(accuracy_score(y, yhat))
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {
        "auc": auc,
        "logloss": ll,
        "acc": acc,
        "prec": float(pr),
        "rec": float(rc),
        "f1": float(f1),
    }


def sample_minibatch(X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    k = min(batch_size, n)
    idx = np.random.choice(n, size=k, replace=False)
    return X[idx], y[idx]


# -------------------------
# Quantum model
# -------------------------
@dataclass
class VQCConfig:
    n_qubits: int = 8
    layers: int = 3
    no_entangle: bool = False
    input_scale: float = 1.0
    logit_scale: float = 2.0
    logit_bias: float = 0.0


class VQCModel:
    def __init__(self, cfg: VQCConfig):
        self.cfg = cfg
        self.n_qubits = cfg.n_qubits

        # (layers, qubits, 3) trainable angles
        self.theta = np.random.uniform(-0.05, 0.05, size=(cfg.layers, cfg.n_qubits, 3)).astype(np.float32)

        # Mean-Z measurement operators across all qubits
        self._z_ops = []
        for q in range(self.n_qubits):
            s = ["I"] * self.n_qubits
            # little-endian: qubit 0 corresponds to rightmost Pauli character
            s[self.n_qubits - 1 - q] = "Z"
            self._z_ops.append(Pauli("".join(s)))

    def _build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)

        xs = (self.cfg.input_scale * x).astype(np.float32)

        # Data encoding
        for i in range(self.n_qubits):
            qc.ry(float(xs[i]), i)
            qc.rz(float(0.5 * xs[i]), i)

        # Trainable layers
        for l in range(self.cfg.layers):
            for q in range(self.n_qubits):
                a, b, c = self.theta[l, q]
                qc.rx(float(a), q)
                qc.ry(float(b), q)
                qc.rz(float(c), q)

            if not self.cfg.no_entangle:
                edges = [
                    (0, 1), (1, 2), (2, 3),
                    (4, 5), (5, 6), (6, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ]
                for a, b in edges:
                    qc.cx(a, b)

        return qc

    def forward_expectation(self, x: np.ndarray) -> float:
        qc = self._build_circuit(x)
        sv = Statevector.from_instruction(qc)
        exps = [np.real(sv.expectation_value(op)) for op in self._z_ops]
        return float(np.mean(exps))

    def predict_proba(self, X: np.ndarray, batch: int = 256) -> np.ndarray:
        out = np.zeros((X.shape[0],), dtype=np.float32)
        for i in range(0, X.shape[0], batch):
            xb = X[i:i + batch]
            e = np.array([self.forward_expectation(x) for x in xb], dtype=np.float32)
            logit = self.cfg.logit_scale * e + self.cfg.logit_bias
            out[i:i + batch] = sigmoid(logit)
        return out


# -------------------------
# SPSA
# -------------------------
def spsa_step(
    model: VQCModel,
    Xb: np.ndarray,
    yb: np.ndarray,
    a: float,
    c: float,
    t: int,
) -> Tuple[float, float]:
    a_t = a / ((t + 1) ** 0.602)
    c_t = c / ((t + 1) ** 0.101)

    delta = np.random.choice([-1.0, 1.0], size=model.theta.shape).astype(np.float32)
    theta0 = model.theta.copy()

    model.theta = (theta0 + c_t * delta).astype(np.float32)
    p_plus = model.predict_proba(Xb, batch=min(256, len(Xb)))
    L_plus = float(log_loss(yb, p_plus, labels=[0, 1]))

    model.theta = (theta0 - c_t * delta).astype(np.float32)
    p_minus = model.predict_proba(Xb, batch=min(256, len(Xb)))
    L_minus = float(log_loss(yb, p_minus, labels=[0, 1]))

    ghat = (L_plus - L_minus) / (2.0 * c_t) * delta
    model.theta = (theta0 - a_t * ghat).astype(np.float32)

    loss_est = 0.5 * (L_plus + L_minus)
    grad_norm = float(np.linalg.norm(ghat))
    return loss_est, grad_norm


# -------------------------
# IO + CLI
# -------------------------
def load_split(
    path: Path,
    label_col: str,
    feature_col: str,
    max_n: int,
    subset_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)

    if max_n is not None and max_n > 0 and len(df) > max_n:
        # IMPORTANT: keep subset sampling FIXED across training seeds
        df = df.sample(n=max_n, random_state=subset_seed).reset_index(drop=True)

    X = get_feature_matrix(df, feature_col=feature_col)
    y = get_labels(df, label_col=label_col)
    return X, y


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train VQC on 8D feature vectors (statevector sim + SPSA).")

    ap.add_argument(
        "--quantum_dir",
        type=str,
        default="data/processed/quantum/sen1floods11_handlabeled",
        help="Directory with train_quantum.parquet/val_quantum.parquet/test_quantum.parquet",
    )
    ap.add_argument("--label_col", type=str, default="y")

    ap.add_argument(
        "--feature_col",
        type=str,
        default="z_hw",
        help="Vector column to use (e.g., z_hw for PCA8, theta_region8 for region8 angles).",
    )

    ap.add_argument("--seed", type=int, default=7, help="Training randomness seed (init + SPSA + minibatches).")
    ap.add_argument(
        "--subset_seed",
        type=int,
        default=0,
        help="Seed used ONLY for sampling the max_train/max_val/max_test subsets. Keep fixed across seeds.",
    )

    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--no_entangle", action="store_true")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--max_val", type=int, default=2000)
    ap.add_argument("--max_test", type=int, default=2000)

    ap.add_argument("--spsa_a", type=float, default=0.15)
    ap.add_argument("--spsa_c", type=float, default=0.07)

    ap.add_argument("--logit_scale", type=float, default=2.0)
    ap.add_argument("--logit_bias", type=float, default=0.0)
    ap.add_argument("--input_scale", type=float, default=1.0)

    ap.add_argument("--print_every", type=int, default=10)

    ap.add_argument("--save_preds", action="store_true", help="Also save val/test predicted probabilities.")
    ap.add_argument("--list_cols", action="store_true", help="Print parquet columns and exit.")

    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/vqc_region8_sim",
        help="Base directory for saving run artifacts.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    qdir = Path(args.quantum_dir)
    tr_path = qdir / "train_quantum.parquet"
    va_path = qdir / "val_quantum.parquet"
    te_path = qdir / "test_quantum.parquet"

    if not tr_path.exists() or not va_path.exists() or not te_path.exists():
        raise FileNotFoundError(
            f"Missing parquet(s). Expected:\n  {tr_path}\n  {va_path}\n  {te_path}"
        )

    if args.list_cols:
        df = pd.read_parquet(tr_path)
        print("Columns:", df.columns.tolist())
        return

    print("Loading splits...")
    Xtr, ytr = load_split(tr_path, args.label_col, args.feature_col, args.max_train, args.subset_seed)
    Xva, yva = load_split(va_path, args.label_col, args.feature_col, args.max_val, args.subset_seed)
    Xte, yte = load_split(te_path, args.label_col, args.feature_col, args.max_test, args.subset_seed)

    if Xtr.shape[1] != 8:
        raise ValueError(
            f"Expected 8D features for 8-qubit VQC, got {Xtr.shape[1]} from feature_col='{args.feature_col}'."
        )

    Xtr, Xva, Xte = standardize_train_only(Xtr, Xva, Xte)

    print(f"Train: {Xtr.shape} | Val: {Xva.shape} | Test: {Xte.shape}")
    uniq, cnt = np.unique(ytr, return_counts=True)
    print(f"Class balance (train): {dict(zip(uniq.tolist(), cnt.tolist()))}")

    cfg = VQCConfig(
        n_qubits=8,
        layers=args.layers,
        no_entangle=args.no_entangle,
        input_scale=args.input_scale,
        logit_scale=args.logit_scale,
        logit_bias=args.logit_bias,
    )
    model = VQCModel(cfg)

    print("\nTraining (SPSA)...")
    t0 = time.time()

    # store last val metrics for quick visibility
    last_val: Optional[Dict[str, float]] = None

    for step in range(1, args.steps + 1):
        Xb, yb = sample_minibatch(Xtr, ytr, args.batch_size)
        train_loss_est, grad_norm = spsa_step(model, Xb, yb, a=args.spsa_a, c=args.spsa_c, t=step)

        if step == 1 or (step % args.print_every == 0) or (step == args.steps):
            pva = model.predict_proba(Xva, batch=256)
            last_val = eval_metrics(yva, pva, thr=0.5)
            print(
                f"step={step:4d}  trainL~{train_loss_est:.4f}  | "
                f"valL={last_val['logloss']:.4f}  val_auc={last_val['auc']:.4f}  "
                f"val_acc@0.5={last_val['acc']:.4f}  val_f1@0.5={last_val['f1']:.4f}  "
                f"|| grad~{grad_norm:.3f}"
            )

    dt = time.time() - t0
    print(f"\nDone. Training time: {dt:.1f}s")

    # Calibrate threshold on VAL (maximize F1)
    pva = model.predict_proba(Xva, batch=256)
    best_thr, best_f1 = best_threshold_by_f1(yva, pva, steps=401)
    mva = eval_metrics(yva, pva, thr=best_thr)

    print("\nVAL (best-threshold calibrated on VAL):")
    print(f"  best_thr={best_thr:.3f}  best_f1={best_f1:.4f}")
    print(
        f"  AUC={mva['auc']:.4f}  logloss={mva['logloss']:.4f}  "
        f"acc={mva['acc']:.4f}  prec={mva['prec']:.4f}  rec={mva['rec']:.4f}  f1={mva['f1']:.4f}"
    )

    # Evaluate on TEST using VAL-calibrated threshold
    pte = model.predict_proba(Xte, batch=256)
    mte = eval_metrics(yte, pte, thr=best_thr)

    print("\nTEST (using VAL-calibrated threshold):")
    print(f"  thr={best_thr:.3f}")
    print(
        f"  AUC={mte['auc']:.4f}  logloss={mte['logloss']:.4f}  "
        f"acc={mte['acc']:.4f}  prec={mte['prec']:.4f}  rec={mte['rec']:.4f}  f1={mte['f1']:.4f}"
    )

    # -------------------------
    # Save run artifacts
    # -------------------------
    run_name = f"{args.feature_col}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save trained weights
    np.save(run_dir / "theta.npy", model.theta)

    # Save args for exact reproducibility
    with open(run_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save metrics
    metrics = {
        "val_best_thr": best_thr,
        "val_best_f1": best_f1,
        "val_metrics": mva,
        "test_metrics": mte,
        "train_time_sec": dt,
        "final_val_at_0.5": last_val,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Optionally save predictions for debugging / plots
    if args.save_preds:
        np.save(run_dir / "val_proba.npy", pva.astype(np.float32))
        np.save(run_dir / "test_proba.npy", pte.astype(np.float32))
        np.save(run_dir / "val_y.npy", yva.astype(np.int64))
        np.save(run_dir / "test_y.npy", yte.astype(np.int64))

    print(f"✅ Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()