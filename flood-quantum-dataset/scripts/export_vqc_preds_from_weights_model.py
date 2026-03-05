#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Export VQC predictions using VQCModel + weights/params npy.")
    ap.add_argument("--vqc_module", required=True, help="e.g., src.models.vqc_hw_sim or src.models.vqc_sim")
    ap.add_argument("--quantum_parquet", required=True, help="Parquet containing patch_id,y and feature_col")
    ap.add_argument("--feature_col", required=True, help="e.g., theta_hw, theta_sim, theta_region8")
    ap.add_argument("--weights_npy", required=True, help="weights.npy or best_params.npy")
    ap.add_argument("--layers", type=int, required=True, help="layers used in run")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--auto_flip", action="store_true", help="If AUC<0.5 on full set, flip p->1-p before saving")
    return ap.parse_args()


def as_matrix(series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=float).reshape(-1) for v in series.to_list()], axis=0)


def find_model_predict_fn(model) -> Callable:
    for name in ["predict_proba", "predict", "forward", "infer", "batch_predict", "run"]:
        if hasattr(model, name) and callable(getattr(model, name)):
            return getattr(model, name)
    methods = [n for n in dir(model) if callable(getattr(model, n)) and not n.startswith("_")]
    raise RuntimeError("VQCModel has no obvious predict method. Public callables:\n" + "\n".join(methods))


def call_predict(fn: Callable, model, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Try common signatures
    for args in [(X, weights), (weights, X)]:
        try:
            return np.asarray(fn(*args))
        except Exception:
            pass

    # Try stuffing weights inside model
    for attr in ["weights", "params", "theta", "w"]:
        try:
            setattr(model, attr, weights)
            return np.asarray(fn(X))
        except Exception:
            pass

    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        pass
    raise RuntimeError(f"Could not call {fn} with signature {sig}")


def normalize_to_prob1(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] == 2:
        return pred[:, 1]
    if pred.ndim == 1:
        # If logits, squash
        if np.nanmin(pred) < 0.0 or np.nanmax(pred) > 1.0:
            return 1.0 / (1.0 + np.exp(-pred))
        return pred
    raise ValueError(f"Unexpected prediction shape: {pred.shape}")


def main():
    args = parse_args()
    mod = importlib.import_module(args.vqc_module)

    if not hasattr(mod, "VQCConfig") or not hasattr(mod, "VQCModel"):
        raise RuntimeError(f"{args.vqc_module} must expose VQCConfig and VQCModel.")

    qpath = Path(args.quantum_parquet)
    wpath = Path(args.weights_npy)
    if not qpath.exists():
        raise FileNotFoundError(qpath)
    if not wpath.exists():
        raise FileNotFoundError(wpath)

    outpath = Path(args.out_csv)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(qpath)
    need = ["patch_id", "y", args.feature_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {qpath}. Columns={df.columns.tolist()}")

    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    X = as_matrix(df[args.feature_col])
    y = df["y"].to_numpy().astype(int)
    n_qubits = X.shape[1]

    weights = np.load(wpath).reshape(-1)

    cfg = mod.VQCConfig(n_qubits=n_qubits, layers=args.layers)
    model = mod.VQCModel(cfg)

    pred_fn = find_model_predict_fn(model)
    print("Using VQCModel method:", pred_fn.__name__)

    raw = call_predict(pred_fn, model, X, weights)
    p = normalize_to_prob1(raw)

    if args.auto_flip:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
        if auc < 0.5:
            print(f"Auto-flip triggered: AUC={auc:.4f} < 0.5. Writing flipped probabilities.")
            p = 1.0 - p

    out = pd.DataFrame({"patch_id": df["patch_id"].to_numpy(), "p_vqc": p})
    out.to_csv(outpath, index=False)
    print(f"✅ wrote {outpath} rows={len(out)} | n_qubits={n_qubits} | layers={args.layers}")


if __name__ == "__main__":
    main()