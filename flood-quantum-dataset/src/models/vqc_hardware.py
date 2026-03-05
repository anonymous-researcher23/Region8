#!/usr/bin/env python3
"""
Run 8-qubit VQC inference on IBM Quantum hardware using Qiskit Runtime SamplerV2
WITHOUT sessions (job mode) — required for OPEN plan accounts.

You trained params on Aer (vqc_hw_sim). Here we run inference on hardware on a small subset.
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime import SamplerV2 as Sampler


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


def load_split(quantum_dir: str, split: str):
    path = os.path.join(quantum_dir, f"{split}_quantum.parquet")
    df = pd.read_parquet(path)

    need = ["theta_hw", "y", "scene_id", "patch_id", "row", "col", "flood_frac"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in {path}. Columns={list(df.columns)}")

    X = np.stack(df["theta_hw"].apply(lambda a: np.asarray(a, dtype=np.float32)).values, axis=0)
    y = df["y"].astype(int).to_numpy()
    meta = df[["scene_id", "patch_id", "row", "col", "flood_frac"]].copy()
    return X, y, meta


def make_vqc_circuit(n_layers: int, x_angles: np.ndarray, w_params: np.ndarray) -> QuantumCircuit:
    n_qubits = 8
    per_layer = 2 * n_qubits

    if len(x_angles) != n_qubits:
        raise ValueError(f"theta_hw must be length 8, got {len(x_angles)}")
    if len(w_params) != n_layers * per_layer:
        raise ValueError(f"w params length mismatch: got {len(w_params)} expected {n_layers*per_layer}")

    qc = QuantumCircuit(n_qubits)
    idx = 0

    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(x_angles[q]), q)

        for q in range(n_qubits):
            qc.ry(float(w_params[idx + q]), q)
        idx += n_qubits

        for q in range(n_qubits):
            qc.rz(float(w_params[idx + q]), q)
        idx += n_qubits

        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)

    qc.measure_all()
    return qc


def probs_from_counts(counts_list, shots, alpha, beta):
    p1 = np.zeros(len(counts_list), dtype=np.float32)
    for i, counts in enumerate(counts_list):
        c0 = 0.0
        c1 = 0.0
        for bitstr, c in counts.items():
            if bitstr[-1] == "0":  # qubit0 (measure_all)
                c0 += c
            else:
                c1 += c
        zexp = (c0 / shots) - (c1 / shots)
        p1[i] = sigmoid(alpha * zexp + beta)
    return p1


def extract_counts_list_from_result(result, n_items: int, shots: int):
    """
    Extract per-circuit counts for SamplerV2 across versions.
    """
    counts_list = []

    for i in range(n_items):
        item = result[i] if hasattr(result, "__getitem__") else result
        counts = None

        # Pattern A: item.data.<reg>.get_counts()
        if hasattr(item, "data"):
            d = item.data
            for attr in dir(d):
                obj = getattr(d, attr, None)
                if obj is None:
                    continue
                if hasattr(obj, "get_counts"):
                    try:
                        counts = obj.get_counts()
                        break
                    except Exception:
                        pass

        # Pattern B: quasi distributions
        if counts is None and hasattr(item, "quasi_dists"):
            qd = item.quasi_dists
            qd0 = qd[0] if isinstance(qd, list) else qd
            counts = {}
            for outcome, p in qd0.items():
                bitstr = format(int(outcome), "08b")
                counts[bitstr] = int(round(float(p) * shots))

        if counts is None:
            raise RuntimeError(
                "Could not extract counts from SamplerV2 result. "
                "Paste `python -c \"import qiskit_ibm_runtime as r; print(r.__version__)\"` "
                "and `python - <<'PY'\\nfrom qiskit_ibm_runtime import QiskitRuntimeService\\nprint(QiskitRuntimeService())\\nPY` if needed."
            )

        counts_list.append(counts)

    return counts_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantum_dir", required=True)
    ap.add_argument("--params_path", required=True)
    ap.add_argument("--out_dir", default="outputs/vqc_hardware")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--subset", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    theta = np.load(args.params_path).astype(np.float32)
    w = theta[:-2]
    alpha = float(theta[-2])
    beta = float(theta[-1])

    X, y, meta = load_split(args.quantum_dir, args.split)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=min(args.subset, len(X)), replace=False)
    Xs = X[idx]
    ys = y[idx]
    metas = meta.iloc[idx].reset_index(drop=True)

    circuits = [make_vqc_circuit(args.n_layers, Xs[i], w) for i in range(len(Xs))]

    service = QiskitRuntimeService()

    if args.backend:
        backend = service.backend(args.backend)
    else:
        candidates = service.backends(operational=True, simulator=False, min_num_qubits=8)
        if not candidates:
            raise RuntimeError("No operational IBM backends with >=8 qubits available to your account.")
        candidates = sorted(candidates, key=lambda b: b.status().pending_jobs)
        backend = candidates[0]

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    # JOB MODE: no Session (open plan compatible)
    sampler = SamplerV2(mode=backend)

    # Optional: set resilience if available
    try:
        sampler.options.resilience_level = 1
    except Exception:
        pass

    tcircs = transpile(circuits, backend=backend, optimization_level=1)

    job = sampler.run(tcircs, shots=args.shots)
    result = job.result()

    dt = time.time() - t0

    counts_list = extract_counts_list_from_result(result, n_items=len(circuits), shots=args.shots)
    p1 = probs_from_counts(counts_list, args.shots, alpha, beta)
    m = compute_metrics(ys, p1, thr=args.thr)

    payload = {
        "backend": backend.name,
        "shots": args.shots,
        "split": args.split,
        "subset": int(len(Xs)),
        "n_layers": args.n_layers,
        "thr": args.thr,
        "alpha": alpha,
        "beta": beta,
        "wall_time_sec": float(dt),
        "metrics": m,
        "notes": "OPEN plan: job-mode SamplerV2 (no Session). Inference-only hardware run.",
    }

    metas["y_true"] = ys
    metas["p1"] = p1
    metas.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    with open(os.path.join(out_dir, "counts.json"), "w") as f:
        json.dump(counts_list, f)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    print("Backend:", backend.name)
    print("Metrics:", json.dumps(m, indent=2))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
