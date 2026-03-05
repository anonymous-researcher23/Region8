#!/usr/bin/env python3
"""
Export per-patch VQC probabilities to CSV for plotting panels.

This version avoids guessing "run_dir" names and instead accepts:
  --params_npy path/to/best_params.npy

It can work with different VQC modules by passing:
  --vqc_module src.models.vqc_region8_sim   (your current statevector trainer)
or:
  --vqc_module src.models.vqc_sim           (your older 16D trainer)
or:
  --vqc_module src.models.vqc_hw_sim        (your older 8D hw-sim trainer)

Assumptions (robust but honest):
- The module provides a class VQCModel, or a function to build it.
- We try the most common APIs:
    model = VQCModel(...)
    probs = model.predict_proba(theta, X)
  If that doesn't exist, we print helpful info about what *does* exist.

Output CSV columns:
  patch_id, p_vqc
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser("Export VQC probabilities per patch_id.")

    ap.add_argument("--quantum_parquet", type=str, required=True,
                    help="Path to split parquet (e.g., test_quantum.parquet)")
    ap.add_argument("--feature_col", type=str, required=True,
                    help="Vector column (e.g., z_hw, z_sim, theta_region8)")
    ap.add_argument("--params_npy", type=str, required=True,
                    help="Path to best_params.npy (or similar)")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Where to write patch_id,p_vqc CSV")

    ap.add_argument("--vqc_module", type=str, default="src.models.vqc_region8_sim",
                    help="Python module that defines the VQC model (e.g., src.models.vqc_sim)")

    # must match training hyperparams
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--no_entangle", action="store_true")
    ap.add_argument("--input_scale", type=float, default=1.0)
    ap.add_argument("--logit_scale", type=float, default=0.0)
    ap.add_argument("--logit_bias", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=256)

    return ap.parse_args()


def load_X_patch_ids(parquet_path: Path, feature_col: str):
    df = pd.read_parquet(parquet_path)
    if "patch_id" not in df.columns:
        raise ValueError(f"'patch_id' not found in {parquet_path}. Columns={df.columns.tolist()}")
    if feature_col not in df.columns:
        raise ValueError(
            f"feature_col='{feature_col}' not found in {parquet_path}. "
            f"Columns={df.columns.tolist()}"
        )

    X = np.vstack([np.asarray(v, dtype=np.float32) for v in df[feature_col].values])
    patch_ids = df["patch_id"].astype(str).values
    return X, patch_ids


def main():
    args = parse_args()

    qpath = Path(args.quantum_parquet)
    ppath = Path(args.params_npy)
    out_csv = Path(args.out_csv)

    if not qpath.exists():
        raise FileNotFoundError(qpath)
    if not ppath.exists():
        raise FileNotFoundError(ppath)

    theta = np.load(ppath)
    X, patch_ids = load_X_patch_ids(qpath, args.feature_col)
    n_qubits = int(X.shape[1])

    mod = importlib.import_module(args.vqc_module)

    # Try to construct model
    if hasattr(mod, "VQCModel"):
        model = mod.VQCModel(
            n_qubits=n_qubits,
            layers=args.layers,
            no_entangle=args.no_entangle,
            input_scale=args.input_scale,
            logit_scale=args.logit_scale,
            logit_bias=args.logit_bias,
        )
    elif hasattr(mod, "build_model"):
        model = mod.build_model(
            n_qubits=n_qubits,
            layers=args.layers,
            no_entangle=args.no_entangle,
            input_scale=args.input_scale,
            logit_scale=args.logit_scale,
            logit_bias=args.logit_bias,
        )
    else:
        raise RuntimeError(
            f"{args.vqc_module} does not expose VQCModel or build_model.\n"
            f"Available symbols include: {sorted([x for x in dir(mod) if 'vqc' in x.lower() or 'model' in x.lower()])}"
        )

    # Try to get probabilities
    if not hasattr(model, "predict_proba"):
        # Print something useful and fail loudly
        candidates = [x for x in dir(model) if "pred" in x.lower() or "prob" in x.lower() or "forward" in x.lower()]
        raise RuntimeError(
            f"Model from {args.vqc_module} has no predict_proba(theta, X).\n"
            f"Found possible methods: {candidates}\n"
            f"Tip: if your model exposes logits/expectations, we can adapt exporter in 1 minute."
        )

    bs = int(args.batch_size)
    probs = []
    for i in range(0, len(X), bs):
        xb = X[i:i+bs]
        pb = model.predict_proba(theta, xb)
        pb = np.asarray(pb, dtype=np.float32).reshape(-1)
        probs.append(pb)

    p = np.concatenate(probs, axis=0)

    out = pd.DataFrame({"patch_id": patch_ids, "p_vqc": p})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}  (n={len(out)})")


if __name__ == "__main__":
    main()