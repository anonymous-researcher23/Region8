#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="src.models.vqc_region8_sim", help="Region8 VQC module")
    ap.add_argument("--weights_npy", required=True, help="Path to weights.npy from region8 run")
    ap.add_argument("--out_csv", required=True, help="Output CSV with patch_id,p_vqc")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to export")
    ap.add_argument("--max_rows", type=int, default=0, help="Limit rows (0 = no limit)")
    ap.add_argument("--shots", type=int, default=1024, help="Shots per circuit")
    ap.add_argument("--batch_size", type=int, default=64, help="Circuits per batch")
    return ap.parse_args()


def get_backend():
    try:
        from qiskit_aer import Aer
        from qiskit import transpile
        backend = Aer.get_backend("aer_simulator")
        return backend, transpile, None, "aer"
    except Exception:
        pass

    try:
        from qiskit import BasicAer, transpile, execute
        backend = BasicAer.get_backend("qasm_simulator")
        return backend, transpile, execute, "basic"
    except Exception as e:
        raise RuntimeError("No simulator backend found. Install qiskit-aer.") from e


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def prob_last_bit_1(counts: dict, shots: int) -> float:
    if not counts:
        return float("nan")
    tot = sum(counts.values()) or shots
    s = 0
    for k, c in counts.items():
        k = k.replace(" ", "")
        if k and k[-1] == "1":
            s += c
    return s / tot


def main():
    args = parse_args()

    mod = importlib.import_module(args.module)

    w_path = Path(args.weights_npy)
    if not w_path.exists():
        raise FileNotFoundError(w_path)
    weights = np.load(w_path).reshape(-1)

    # ---- Load data the SAME way the module does ----
    # We support two common patterns:
    # 1) module has a function load_splits() returning (Xtr,ytr,meta_tr, Xva,yva,meta_va, Xte,yte,meta_te)
    # 2) module has a function load_split(split) returning (X,y,meta) where meta has patch_id
    if hasattr(mod, "load_split"):
        X, y, meta = mod.load_split(args.split)
    elif hasattr(mod, "load_splits"):
        out = mod.load_splits()
        # Try to interpret output
        # expected could be: (Xtr,ytr, Xva,yva, Xte,yte, meta_tr,meta_va,meta_te) or similar.
        # We'll do a safe approach: look for tuples by size.
        raise RuntimeError(
            "Module has load_splits() but not load_split(). "
            "Please add a simple load_split(split) wrapper in src/models/vqc_region8_sim.py."
        )
    else:
        raise RuntimeError(
            "src.models.vqc_region8_sim must expose load_split(split) returning (X,y,meta) "
            "where meta includes patch_id."
        )

    if args.max_rows and args.max_rows > 0:
        X = X[: args.max_rows]
        y = y[: args.max_rows]
        meta = meta.iloc[: args.max_rows].copy()

    if "patch_id" not in meta.columns:
        raise RuntimeError("meta returned by load_split() must contain 'patch_id'")

    # ---- Build circuits ----
    if not hasattr(mod, "make_vqc_circuit"):
        raise RuntimeError("Module must expose make_vqc_circuit(...)")

    make_fn = mod.make_vqc_circuit
    sig = inspect.signature(make_fn)
    names = list(sig.parameters.keys())

    # Common patterns:
    # (x_angles, params)  OR (n_qubits,n_layers,x_angles,params)
    circuits = []
    for xx in X:
        xx = np.asarray(xx, dtype=float).reshape(-1)

        try:
            if names == ["x_angles", "params"]:
                qc = make_fn(xx, weights)
            elif names == ["n_qubits", "n_layers", "x_angles", "params"]:
                # infer n_qubits from x vector, infer n_layers by probing
                n_qubits = len(xx)

                # probe layer count 1..20
                n_layers = None
                for L in range(1, 21):
                    try:
                        _ = make_fn(n_qubits, L, xx, weights)
                        n_layers = L
                        break
                    except Exception:
                        continue
                if n_layers is None:
                    raise RuntimeError("Could not infer n_layers for region8 VQC circuit (probe failed).")

                qc = make_fn(n_qubits, n_layers, xx, weights)
            else:
                # last resort try canonical 2-arg
                qc = make_fn(xx, weights)
        except Exception as e:
            raise RuntimeError(f"Failed building circuit with signature {sig}: {e}") from e

        circuits.append(qc)

    # ---- Simulate ----
    backend, transpile_fn, execute_fn, kind = get_backend()
    probs = []

    for batch in chunks(circuits, args.batch_size):
        tqc = transpile_fn(batch, backend=backend) if "backend" in inspect.signature(transpile_fn).parameters else transpile_fn(batch, backend)
        if kind == "aer":
            job = backend.run(tqc, shots=args.shots)
            res = job.result()
        else:
            job = execute_fn(tqc, backend=backend, shots=args.shots)
            res = job.result()

        for i in range(len(batch)):
            probs.append(prob_last_bit_1(res.get_counts(i), args.shots))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"patch_id": meta["patch_id"].to_numpy(), "p_vqc": np.asarray(probs)})
    out.to_csv(out_path, index=False)
    print(f"✅ wrote {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()