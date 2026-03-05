#!/usr/bin/env python3
"""
Train a VQC (statevector simulation) on spatial Region8 features for flood / non-flood.

Input data:
  Parquet splits in:
    data/processed/quantum_region8/sen1floods11_handlabeled/
      train_quantum.parquet
      val_quantum.parquet
      test_quantum.parquet

Required columns:
  - theta_region8 : array-like length 8 (angles in ~[-pi, pi])
  - y             : 0/1 label

What this script does:
  1) Loads train/val/test
  2) Builds a spatially-entangled 8-qubit circuit:
       q0 q1
       q2 q3
       q4 q5
       q6 q7
     with horizontal + vertical entanglement
  3) Uses data re-uploading:
     apply RY(theta_i) before each ansatz layer
  4) Trains weights using SPSA (gradient-free, NISQ-friendly)
  5) Evaluates on val/test and saves artifacts

Practical note:
  - Statevector sim + big dataset is slow.
  - Start with --max_train and --steps to validate end-to-end first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Pauli

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# -----------------------------
# Circuit: Spatial ansatz (8q)
# -----------------------------
def build_spatial_ansatz_8(n_layers: int = 2) -> Tuple[QuantumCircuit, ParameterVector]:
    """
    Parameterized spatial ansatz for 8 qubits laid out as 4x2 regions:
        q0 q1
        q2 q3
        q4 q5
        q6 q7

    Entanglement per layer:
      horizontal: (0-1)(2-3)(4-5)(6-7)
      vertical:   (0-2-4-6) and (1-3-5-7)

    Rotations per qubit per layer: RX, RY, RZ (3 params)
    """
    n_qubits = 8
    qc = QuantumCircuit(n_qubits)
    params = ParameterVector("w", length=n_layers * n_qubits * 3)

    k = 0
    for _ in range(n_layers):
        # local trainable rotations  ✅ FIXED: include qubit index
        for q in range(n_qubits):
            qc.rx(params[k], q); k += 1
            qc.ry(params[k], q); k += 1
            qc.rz(params[k], q); k += 1

        # spatial entanglement
        # horizontal pairs
        qc.cx(0, 1); qc.cx(2, 3); qc.cx(4, 5); qc.cx(6, 7)
        # vertical left column
        qc.cx(0, 2); qc.cx(2, 4); qc.cx(4, 6)
        # vertical right column
        qc.cx(1, 3); qc.cx(3, 5); qc.cx(5, 7)

    return qc, params


def build_data_reuploading_vqc(n_layers: int = 2) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Full VQC circuit = [data-encoding] + [trainable ansatz], repeated per layer:
      For each layer:
        - apply RY(theta_i) on each qubit (data re-upload)
        - apply trainable spatial ansatz layer

    Returns:
      qc        : parameterized QuantumCircuit
      w_params  : trainable weight parameters
      x_params  : input data parameters (theta_0..theta_7)
    """
    n_qubits = 8
    x_params = ParameterVector("x", length=n_qubits)

    # We only need w_params length/order; circuit is constructed below.
    _, w_params = build_spatial_ansatz_8(n_layers=n_layers)

    qc = QuantumCircuit(n_qubits)
    k = 0
    for _ in range(n_layers):
        # data upload
        for q in range(n_qubits):
            qc.ry(x_params[q], q)

        # trainable rotations
        for q in range(n_qubits):
            qc.rx(w_params[k], q); k += 1
            qc.ry(w_params[k], q); k += 1
            qc.rz(w_params[k], q); k += 1

        # entanglement
        qc.cx(0, 1); qc.cx(2, 3); qc.cx(4, 5); qc.cx(6, 7)
        qc.cx(0, 2); qc.cx(2, 4); qc.cx(4, 6)
        qc.cx(1, 3); qc.cx(3, 5); qc.cx(5, 7)

    return qc, w_params, x_params


# -----------------------------
# Model: expectation -> prob
# -----------------------------
@dataclass
class VQCModel:
    qc: QuantumCircuit
    w_params: ParameterVector
    x_params: ParameterVector
    n_qubits: int = 8

    def _bind(self, x: np.ndarray, w: np.ndarray) -> QuantumCircuit:
        bind_dict = {}
        for i, p in enumerate(self.x_params):
            bind_dict[p] = float(x[i])
        for i, p in enumerate(self.w_params):
            bind_dict[p] = float(w[i])
        return self.qc.assign_parameters(bind_dict, inplace=False)

    def forward_expectation(self, x: np.ndarray, w: np.ndarray) -> float:
        """
        Compute expectation value of Z on qubit 0 (range [-1, 1]).
        NOTE: Qiskit Pauli strings are little-endian: rightmost char is qubit 0.
        """
        bound = self._bind(x, w)
        sv = Statevector.from_instruction(bound)

        # Measure Z on qubit 0 => "IIIIIIIZ" (rightmost is qubit 0)
        op = Pauli("I" * (self.n_qubits - 1) + "Z")
        exp = np.real(sv.expectation_value(op))
        return float(exp)

    def predict_proba(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Convert expectation in [-1,1] -> probability in [0,1].
        p = (1 + exp)/2
        """
        exps = np.array([self.forward_expectation(x, w) for x in X], dtype=np.float32)
        p = (1.0 + exps) * 0.5
        return np.clip(p, 1e-6, 1 - 1e-6)


# -----------------------------
# Loss + Optimizer (SPSA)
# -----------------------------
def binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def spsa_step(
    model: VQCModel,
    Xb: np.ndarray,
    yb: np.ndarray,
    w: np.ndarray,
    a: float,
    c: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    One SPSA update:
      ghat ≈ (L(w + c*Δ) - L(w - c*Δ)) / (2c) * Δ^{-1}
      w <- w - a * ghat
    """
    delta = rng.choice([-1.0, 1.0], size=w.shape).astype(np.float32)

    w_plus = w + c * delta
    w_minus = w - c * delta

    p_plus = model.predict_proba(Xb, w_plus)
    p_minus = model.predict_proba(Xb, w_minus)

    L_plus = binary_log_loss(p_plus, yb)
    L_minus = binary_log_loss(p_minus, yb)

    ghat = (L_plus - L_minus) / (2.0 * c) * (1.0 / delta)
    w_new = w - a * ghat

    info = {"L_plus": L_plus, "L_minus": L_minus, "L_mid": 0.5 * (L_plus + L_minus)}
    return w_new.astype(np.float32), info


# -----------------------------
# Data utilities
# -----------------------------
def load_split(parquet_path: Path, feature_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    if feature_col not in df.columns:
        raise ValueError(f"Missing '{feature_col}' in {parquet_path}. Columns: {list(df.columns)}")
    if "y" not in df.columns:
        raise ValueError(f"Missing 'y' in {parquet_path}. Columns: {list(df.columns)}")

    X = np.vstack(df[feature_col].to_numpy()).astype(np.float32)
    y = df["y"].to_numpy().astype(np.int64)

    if X.shape[1] != 8:
        raise ValueError(f"Expected 8 features, got {X.shape} from {parquet_path}")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values in {parquet_path} ({feature_col}).")

    return X, y


def subsample(X: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_n <= 0 or max_n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_n, replace=False)
    return X[idx], y[idx]


# -----------------------------
# Metrics
# -----------------------------
def eval_metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    y_pred = (p >= thr).astype(np.int64)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, p)),
        "thr": float(thr),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=4),
    }
    return out


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train spatial VQC (statevector sim) on theta_region8 features.")
    ap.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/quantum_region8/sen1floods11_handlabeled",
        help="Folder containing train/val/test_quantum.parquet",
    )
    ap.add_argument("--feature_col", type=str, default="theta_region8", help="Feature column to use")
    ap.add_argument("--out_dir", type=str, default="outputs/vqc_region8_sim", help="Output directory base")
    ap.add_argument("--layers", type=int, default=2, help="Number of VQC layers")
    ap.add_argument("--steps", type=int, default=200, help="SPSA steps")
    ap.add_argument("--batch_size", type=int, default=128, help="Mini-batch size for SPSA loss estimates")
    ap.add_argument(
        "--max_train",
        type=int,
        default=4000,
        help="Subsample training set for speed (0 = use full train; not recommended for statevector)",
    )
    ap.add_argument("--max_val", type=int, default=2000, help="Subsample val set for quick eval (0 = full)")
    ap.add_argument("--max_test", type=int, default=4000, help="Subsample test set for quick eval (0 = full)")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for metrics")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--a0", type=float, default=0.05, help="SPSA step-size base a0")
    ap.add_argument("--c0", type=float, default=0.10, help="SPSA perturbation base c0")
    ap.add_argument("--a_decay", type=float, default=0.101, help="SPSA a_k decay exponent (common ~0.101)")
    ap.add_argument("--c_decay", type=float, default=0.101, help="SPSA c_k decay exponent (common ~0.101)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    train_p = data_dir / "train_quantum.parquet"
    val_p = data_dir / "val_quantum.parquet"
    test_p = data_dir / "test_quantum.parquet"

    if not train_p.exists() or not val_p.exists() or not test_p.exists():
        raise FileNotFoundError(f"Missing parquet splits in {data_dir}.")

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading splits...")
    Xtr, ytr = load_split(train_p, args.feature_col)
    Xva, yva = load_split(val_p, args.feature_col)
    Xte, yte = load_split(test_p, args.feature_col)

    Xtr, ytr = subsample(Xtr, ytr, args.max_train, args.seed)
    Xva, yva = subsample(Xva, yva, args.max_val, args.seed + 1)
    Xte, yte = subsample(Xte, yte, args.max_test, args.seed + 2)

    print(f"Train: {Xtr.shape} | Val: {Xva.shape} | Test: {Xte.shape}")
    print(f"Class balance (train): {{0:{int((ytr==0).sum())}, 1:{int((ytr==1).sum())}}}")

    qc, w_params, x_params = build_data_reuploading_vqc(n_layers=args.layers)
    model = VQCModel(qc=qc, w_params=w_params, x_params=x_params)

    n_w = len(w_params)
    w = rng.normal(loc=0.0, scale=0.1, size=(n_w,)).astype(np.float32)

    history: List[Dict[str, Any]] = []
    print("\nTraining (SPSA)...")
    for k in range(1, args.steps + 1):
        idx = rng.choice(len(ytr), size=min(args.batch_size, len(ytr)), replace=False)
        Xb, yb = Xtr[idx], ytr[idx]

        a = args.a0 / (k ** args.a_decay)
        c = args.c0 / (k ** args.c_decay)

        w, info = spsa_step(model, Xb, yb, w, a=a, c=c, rng=rng)

        if k == 1 or k % 10 == 0 or k == args.steps:
            pva = model.predict_proba(Xva, w)
            Lva = binary_log_loss(pva, yva)
            mva = eval_metrics(yva, pva, thr=args.thr)

            rec = {
                "step": k,
                "a": float(a),
                "c": float(c),
                "train_L_mid": float(info["L_mid"]),
                "val_logloss": float(Lva),
                "val_acc": mva["acc"],
                "val_f1": mva["f1"],
                "val_auc": mva["auc"],
            }
            history.append(rec)
            print(
                f"step={k:4d}  trainL~{info['L_mid']:.4f}  valL={Lva:.4f}  "
                f"val_acc={mva['acc']:.4f} val_f1={mva['f1']:.4f} val_auc={mva['auc']:.4f}"
            )

    print("\nFinal evaluation...")
    pva = model.predict_proba(Xva, w)
    pte = model.predict_proba(Xte, w)

    val_metrics = eval_metrics(yva, pva, thr=args.thr)
    test_metrics = eval_metrics(yte, pte, thr=args.thr)

    print("\nVAL:")
    print(f"  acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}")
    print("\nTEST:")
    print(f"  acc={test_metrics['acc']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}")

    np.save(out_dir / "weights.npy", w)

    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save circuit QASM if supported
    try:
        qasm = model.qc.qasm()
        with open(out_dir / "vqc_region8.qasm", "w") as f:
            f.write(qasm)
    except Exception:
        pass

    print(f"\n✅ Saved run to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()