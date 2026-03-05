#!/usr/bin/env python3
"""
VQC simulation training for flood detection using angle-encoded features (theta_sim).

Robust simulator path:
  - Uses qiskit-aer AerSimulator backend (counts-based), NOT primitives Sampler,
    to avoid Qiskit primitive API changes (SamplerV2 result formats differ).

Fixes included:
  - Class-weighted BCE (pos_weight) to handle imbalance (prevents F1=0 collapse)
  - Callback picks best F1 over a small threshold grid (more informative than fixed 0.5)
  - Prints positive counts in sampled subsets

Inputs:
  data/processed/quantum/<dataset_name>/train_quantum.parquet
  data/processed/quantum/<dataset_name>/val_quantum.parquet
  data/processed/quantum/<dataset_name>/test_quantum.parquet

Expected columns:
  - theta_sim: list/array length=16 (angles in radians)
  - y: 0/1 label

Outputs:
  outputs/vqc_sim/<run_name>/{metrics.json, best_params.npy, curves.csv}
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# ------------------ Qiskit imports ------------------

def require_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile
        return QuantumCircuit, transpile
    except Exception as e:
        raise RuntimeError("Qiskit import failed. Try: pip install qiskit") from e

QuantumCircuit, transpile = require_qiskit()

def require_aer_simulator():
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator
    except Exception as e:
        raise RuntimeError("qiskit-aer is required. Try: pip install qiskit-aer") from e

AerSimulator = require_aer_simulator()

def require_spsa():
    try:
        from qiskit_algorithms.optimizers import SPSA
        return SPSA
    except Exception:
        try:
            from qiskit.algorithms.optimizers import SPSA
            return SPSA
        except Exception as e:
            raise RuntimeError("SPSA not found. Try: pip install qiskit-algorithms") from e

SPSA = require_spsa()

# ------------------ utilities ------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_quantum_parquet(path: str):
    df = pd.read_parquet(path)
    if "theta_sim" not in df.columns:
        raise ValueError(f"Missing theta_sim in {path}. Columns={list(df.columns)}")
    if "y" not in df.columns:
        raise ValueError(f"Missing y in {path}. Columns={list(df.columns)}")

    X = np.stack(df["theta_sim"].apply(lambda a: np.asarray(a, dtype=np.float32)).values, axis=0)
    y = df["y"].astype(int).to_numpy()
    return X, y

def compute_metrics(y_true, p1, thr=0.5):
    y_pred = (p1 >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, p1))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": auc,
        "confusion_matrix": cm,
    }

def best_f1_over_thresholds(y_true, p1, thresholds=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5)):
    best = None
    for thr in thresholds:
        m = compute_metrics(y_true, p1, thr=thr)
        if best is None or m["f1"] > best["f1"]:
            best = m
            best["thr"] = float(thr)
    return best

# ------------------ circuit ------------------

def make_vqc_circuit(n_qubits: int, n_layers: int, x_angles: np.ndarray, params: np.ndarray) -> QuantumCircuit:
    """
    Data reuploading VQC:
      For each layer:
        - Encode x via RY(x_i) on each qubit
        - Trainable Ry/Rz on each qubit
        - Entangling ring of CX
    Measure:
      - measure all qubits; we use qubit0 (last bit) to estimate <Z0>
    """
    per_layer = 2 * n_qubits
    if len(params) != n_layers * per_layer:
        raise ValueError(f"params size mismatch: got {len(params)}, expected {n_layers * per_layer}")

    qc = QuantumCircuit(n_qubits)
    idx = 0

    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(x_angles[q]), q)

        for q in range(n_qubits):
            qc.ry(float(params[idx + q]), q)
        idx += n_qubits

        for q in range(n_qubits):
            qc.rz(float(params[idx + q]), q)
        idx += n_qubits

        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)

    qc.measure_all()
    return qc

# ------------------ simulator prediction (counts-based) ------------------

def batch_predict_proba_aer_counts(
    backend,
    X: np.ndarray,
    params: np.ndarray,
    n_layers: int,
    alpha: float,
    beta: float,
    shots: int = 128,
    seed: int = 123,
):
    """
    Runs circuits on AerSimulator and returns P(class=1) using <Z0> from measurement counts.
    <Z0> = p(meas0=0) - p(meas0=1), where meas0 is last bit in bitstring from measure_all.
    prob = sigmoid(alpha * <Z0> + beta)
    """
    n_qubits = X.shape[1]
    circuits = [make_vqc_circuit(n_qubits, n_layers, X[i], params) for i in range(X.shape[0])]

    # transpile as a batch for speed
    tcircs = transpile(circuits, backend=backend, optimization_level=1)

    job = backend.run(tcircs, shots=shots, seed_simulator=seed)
    result = job.result()

    probs = np.zeros(X.shape[0], dtype=np.float32)

    for i in range(X.shape[0]):
        counts = result.get_counts(i)  # dict: bitstring -> count
        p0 = 0.0
        p1 = 0.0
        for bitstr, c in counts.items():
            meas0 = bitstr[-1]  # qubit0 from measure_all
            if meas0 == "0":
                p0 += c
            else:
                p1 += c
        p0 /= shots
        p1 /= shots
        zexp = (p0 - p1)
        probs[i] = sigmoid(alpha * zexp + beta)

    return probs

# ------------------ manual SPSA ------------------

def spsa_manual(
    objective,
    x0,
    iters=50,
    a=0.2,
    c=0.1,
    alpha=0.602,
    gamma=0.101,
    seed=123,
    callback=None,
):
    rng = np.random.default_rng(seed)
    theta = x0.astype(np.float32).copy()

    best_theta = theta.copy()
    best_loss = float("inf")

    for k in range(iters):
        ak = a / ((k + 1) ** alpha)
        ck = c / ((k + 1) ** gamma)

        delta = rng.choice([-1.0, 1.0], size=theta.shape).astype(np.float32)

        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta

        loss_plus = float(objective(theta_plus))
        loss_minus = float(objective(theta_minus))

        ghat = (loss_plus - loss_minus) / (2.0 * ck) * delta
        theta = theta - ak * ghat

        loss = float(objective(theta))

        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()

        if callback is not None:
            callback(k, theta, loss, ak, True)

        if k % 10 == 0:
            print(f"[manual SPSA iter {k:04d}] loss={loss:.4f} step={ak:.5f}")

    class DummyRes:
        def __init__(self, x, fun):
            self.x = x
            self.fun = fun
            self.nfev = -1

    return DummyRes(best_theta, best_loss)

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantum_dir", required=True)
    ap.add_argument("--out_dir", default="outputs/vqc_sim")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--shots", type=int, default=128, help="shots per circuit during training")
    ap.add_argument("--train_subset", type=int, default=128)
    ap.add_argument("--val_subset", type=int, default=256)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    train_path = os.path.join(args.quantum_dir, "train_quantum.parquet")
    val_path   = os.path.join(args.quantum_dir, "val_quantum.parquet")
    test_path  = os.path.join(args.quantum_dir, "test_quantum.parquet")

    Xtr, ytr = load_quantum_parquet(train_path)
    Xva, yva = load_quantum_parquet(val_path)
    Xte, yte = load_quantum_parquet(test_path)

    if Xtr.shape[1] != 16:
        raise ValueError(f"theta_sim must be 16D for VQC sim. Got {Xtr.shape}")

    rng = np.random.default_rng(args.seed)
    tr_idx = rng.choice(len(Xtr), size=min(args.train_subset, len(Xtr)), replace=False)
    va_idx = rng.choice(len(Xva), size=min(args.val_subset, len(Xva)), replace=False)

    Xtr_s, ytr_s = Xtr[tr_idx], ytr[tr_idx]
    Xva_s, yva_s = Xva[va_idx], yva[va_idx]

    print(f"Train subset positives: {int(ytr_s.sum())} / {len(ytr_s)}")
    print(f"Val subset positives:   {int(yva_s.sum())} / {len(yva_s)}")

    # Class weight (pos_weight) computed on TRAIN subset
    pos = max(1, int(ytr_s.sum()))
    neg = max(1, int((1 - ytr_s).sum()))
    pos_weight = float(neg / pos)
    print(f"Using pos_weight={pos_weight:.3f} (neg/pos on train subset)")

    backend = AerSimulator(method="automatic")

    n_qubits = 16
    per_layer = 2 * n_qubits
    n_params = args.n_layers * per_layer
    n_params_total = n_params + 2  # alpha, beta

    params0 = (0.01 * rng.standard_normal(n_params_total)).astype(np.float32)

    best = {"f1": -1.0, "params": params0.copy(), "iter": -1, "thr": 0.5}
    curves = []

    def objective(theta):
        w = theta[:-2]
        alpha = float(theta[-2])
        beta = float(theta[-1])

        p1 = batch_predict_proba_aer_counts(
            backend, Xtr_s, w, args.n_layers, alpha, beta, shots=args.shots, seed=args.seed
        )

        # class-weighted BCE
        eps = 1e-6
        y = ytr_s.astype(np.float32)
        loss = -np.mean(
            pos_weight * y * np.log(p1 + eps) + (1.0 - y) * np.log(1.0 - p1 + eps)
        )
        return float(loss)

    def callback(i, theta, fval, stepsize, accepted):
        w = theta[:-2]
        alpha = float(theta[-2])
        beta = float(theta[-1])

        p1v = batch_predict_proba_aer_counts(
            backend, Xva_s, w, args.n_layers, alpha, beta, shots=args.shots, seed=args.seed + 7
        )

        m_best = best_f1_over_thresholds(yva_s, p1v)
        curves.append({
            "iter": int(i),
            "train_loss": float(fval),
            "val_acc_best": m_best["accuracy"],
            "val_f1_best": m_best["f1"],
            "val_auc": m_best["roc_auc"],
            "best_thr": m_best["thr"],
            "step": float(stepsize) if stepsize is not None else None,
        })

        if m_best["f1"] > best["f1"]:
            best["f1"] = m_best["f1"]
            best["params"] = theta.copy()
            best["iter"] = int(i)
            best["thr"] = float(m_best["thr"])

        if i % 10 == 0:
            print(
                f"[iter {i:04d}] loss={fval:.4f} | "
                f"val_f1(best)={m_best['f1']:.4f} @thr={m_best['thr']:.2f} | "
                f"val_acc={m_best['accuracy']:.4f} | val_auc={m_best['roc_auc']:.4f}"
            )

    # Native SPSA may exist but your version doesn't accept callback; we keep the try anyway
    optimizer = SPSA(maxiter=args.iters)

    t0 = time.time()
    try:
        opt_res = optimizer.minimize(fun=objective, x0=params0, callback=callback)
    except TypeError:
        print("SPSA.minimize() does not support callback in this Qiskit version. Using manual SPSA loop.")
        opt_res = spsa_manual(
            objective, params0, iters=args.iters, a=0.2, c=0.1, seed=args.seed, callback=callback
        )
    dt = time.time() - t0

    theta_best = best["params"]
    w_best = theta_best[:-2]
    alpha_best = float(theta_best[-2])
    beta_best = float(theta_best[-1])

    # Evaluate on full val/test with more shots for stability
    eval_shots = max(args.shots, 512)

    p1_val = batch_predict_proba_aer_counts(
        backend, Xva, w_best, args.n_layers, alpha_best, beta_best, shots=eval_shots, seed=args.seed + 11
    )
    p1_test = batch_predict_proba_aer_counts(
        backend, Xte, w_best, args.n_layers, alpha_best, beta_best, shots=eval_shots, seed=args.seed + 13
    )

    val_m = compute_metrics(yva, p1_val, thr=best["thr"])
    test_m = compute_metrics(yte, p1_test, thr=best["thr"])

    metrics = {
        "backend": "AerSimulator(counts)",
        "n_layers": args.n_layers,
        "shots_train": args.shots,
        "shots_eval": eval_shots,
        "iters": args.iters,
        "train_subset": len(Xtr_s),
        "val_subset": len(Xva_s),
        "pos_weight": pos_weight,
        "best_iter": best["iter"],
        "best_thr": best["thr"],
        "best_val_subset_f1": float(best["f1"]),
        "wall_time_sec": float(dt),
        "val": val_m,
        "test": test_m,
        "opt_result": {
            "fun": float(getattr(opt_res, "fun", float("nan"))),
            "nfev": int(getattr(opt_res, "nfev", -1)),
        },
    }

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "best_params.npy"), theta_best)
    pd.DataFrame(curves).to_csv(os.path.join(out_dir, "curves.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. best_iter={best['iter']} best_val_f1={best['f1']:.4f} thr={best['thr']:.2f} wall_time_sec={dt:.1f}")
    print(f"Saved: {out_dir}")

if __name__ == "__main__":
    main()
