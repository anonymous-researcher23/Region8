#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Export Region8 VQC predictions using VQCModel + weights.npy.")
    ap.add_argument("--vqc_module", default="src.models.vqc_region8_sim")
    ap.add_argument("--quantum_parquet", required=True)
    ap.add_argument("--feature_col", default="theta_region8")
    ap.add_argument("--weights_npy", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_rows", type=int, default=0)
    return ap.parse_args()


def as_matrix(series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=float).reshape(-1) for v in series.to_list()], axis=0)


def find_model_predict_fn(model) -> Callable:
    """
    Find a usable predict function on the model.
    Returns a callable f(X, weights) -> probs (shape [N] or [N,2]).

    We prefer functions that explicitly take (X, weights) or (X, params) etc.
    """
    candidates = [
        "predict_proba",
        "predict",
        "forward",
        "infer",
        "batch_predict",
        "batch_forward",
        "run",
    ]
    for name in candidates:
        if hasattr(model, name) and callable(getattr(model, name)):
            fn = getattr(model, name)
            return fn

    # No obvious method: print help for debugging
    methods = [n for n in dir(model) if callable(getattr(model, n)) and not n.startswith("_")]
    raise RuntimeError(
        "VQCModel has no obvious predict method. Public callables are:\n"
        + "\n".join(methods)
    )


def call_predict(fn: Callable, model, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Try calling predict function with a few common signatures.
    """
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        pass

    # Try common patterns
    for args in [
        (X, weights),
        (X, weights, ),
        (weights, X),
    ]:
        try:
            out = fn(*args)
            return np.asarray(out)
        except Exception:
            pass

    # Some models store weights inside model: try setting attribute then calling with X only
    for attr in ["weights", "params", "theta", "w"]:
        try:
            setattr(model, attr, weights)
            out = fn(X)
            return np.asarray(out)
        except Exception:
            pass

    raise RuntimeError(f"Could not call {fn} with signature {sig}")


def normalize_to_prob1(pred: np.ndarray) -> np.ndarray:
    """
    Convert model output to a 1D probability-like vector for class 1.

    Accepts:
    - shape (N,) already
    - shape (N,2): take [:,1]
    - shape (N,): logits allowed (we sigmoid)
    """
    pred = np.asarray(pred)

    if pred.ndim == 2 and pred.shape[1] == 2:
        p1 = pred[:, 1]
        return p1

    if pred.ndim == 1:
        # If looks like logits (not in [0,1]), squash
        if np.nanmin(pred) < 0.0 or np.nanmax(pred) > 1.0:
            return 1.0 / (1.0 + np.exp(-pred))
        return pred

    raise ValueError(f"Unexpected prediction shape: {pred.shape}")


def main():
    args = parse_args()

    mod = importlib.import_module(args.vqc_module)

    if not hasattr(mod, "VQCConfig") or not hasattr(mod, "VQCModel"):
        raise RuntimeError(f"{args.vqc_module} must expose VQCConfig and VQCModel (it does in your output).")

    qpath = Path(args.quantum_parquet)
    wpath = Path(args.weights_npy)
    if not qpath.exists():
        raise FileNotFoundError(qpath)
    if not wpath.exists():
        raise FileNotFoundError(wpath)

    outpath = Path(args.out_csv)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(qpath)
    if "patch_id" not in df.columns:
        raise ValueError("Parquet must contain patch_id")
    if "y" not in df.columns:
        raise ValueError("Parquet must contain y")
    if args.feature_col not in df.columns:
        raise ValueError(f"Missing {args.feature_col}. Columns={df.columns.tolist()}")

    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    X = as_matrix(df[args.feature_col])
    n_qubits = X.shape[1]

    weights = np.load(wpath).reshape(-1)

    # Build model
    cfg = mod.VQCConfig(n_qubits=n_qubits, layers=args.layers)
    model = mod.VQCModel(cfg)

    # Find predict function
    pred_fn = find_model_predict_fn(model)
    print("Using VQCModel method:", pred_fn.__name__)

    raw = call_predict(pred_fn, model, X, weights)
    p1 = normalize_to_prob1(raw)

    out = pd.DataFrame({"patch_id": df["patch_id"].to_numpy(), "p_vqc": p1})
    out.to_csv(outpath, index=False)
    print(f"✅ wrote {outpath} rows={len(out)} | n_qubits={n_qubits} | layers={args.layers}")


if __name__ == "__main__":
    main()