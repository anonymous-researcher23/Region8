#!/usr/bin/env python3
"""
Export VQC predictions by constructing circuits from a model module's make_vqc_circuit()
and executing them on a local simulator backend.

Your modules:
    make_vqc_circuit(n_qubits: int, n_layers: int, x_angles: np.ndarray, params: np.ndarray) -> QuantumCircuit

Key feature:
- Automatically infers n_layers by probing make_vqc_circuit with candidate layer counts.
  This handles cases where params length is NOT simply n_layers * per_layer (e.g., +2 global params).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Circuit construction helpers
# ----------------------------

def build_circuit(mod, n_qubits: int, n_layers: int, x_angles: np.ndarray, params: np.ndarray):
    """
    Robustly call mod.make_vqc_circuit across possible signatures.

    Preferred signature:
        (n_qubits, n_layers, x_angles, params)
    """
    if not hasattr(mod, "make_vqc_circuit"):
        raise RuntimeError(f"{mod.__name__} does not expose make_vqc_circuit()")

    fn = mod.make_vqc_circuit
    sig = inspect.signature(fn)
    names = list(sig.parameters.keys())

    if names == ["n_qubits", "n_layers", "x_angles", "params"]:
        return fn(n_qubits, n_layers, x_angles, params)

    # Try canonical positional order
    try:
        return fn(n_qubits, n_layers, x_angles, params)
    except TypeError:
        pass

    # Try keyword-based mapping
    kw: Dict[str, Any] = {}
    if "n_qubits" in names: kw["n_qubits"] = n_qubits
    if "n_layers" in names: kw["n_layers"] = n_layers
    if "x_angles" in names: kw["x_angles"] = x_angles
    if "params" in names: kw["params"] = params

    if kw:
        return fn(**kw)

    raise RuntimeError(f"Don't know how to call {mod.__name__}.make_vqc_circuit with signature: {sig}")


def infer_n_layers_by_probe(
    mod,
    n_qubits: int,
    params: np.ndarray,
    max_layers: int = 20,
) -> int:
    """
    Infer n_layers by *probing* mod.make_vqc_circuit with candidate layer counts.

    This is robust to circuits where params length includes extra global parameters
    (e.g., 2*n_qubits*n_layers + 2).

    We try layers=1..max_layers and pick the first one that does NOT raise the
    module's "params size mismatch" ValueError.
    """
    x_dummy = np.zeros(n_qubits, dtype=float)

    last_err: Optional[Exception] = None
    for L in range(1, max_layers + 1):
        try:
            _ = build_circuit(mod, n_qubits=n_qubits, n_layers=L, x_angles=x_dummy, params=params)
            return L
        except ValueError as e:
            # The module raises ValueError("params size mismatch: ...") for wrong L
            msg = str(e).lower()
            if "params size mismatch" in msg or "params" in msg and "expected" in msg:
                last_err = e
                continue
            # Some other ValueError – still record but continue probing
            last_err = e
            continue
        except Exception as e:
            # If it's something else (e.g., qiskit import weirdness), record and keep trying
            last_err = e
            continue

    raise ValueError(
        f"Could not infer n_layers by probing 1..{max_layers}. "
        f"Last error: {last_err}"
    )


# ----------------------------
# Backend + execution helpers
# ----------------------------

def get_backend():
    """Prefer Aer if installed; otherwise fall back to BasicAer."""
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
        raise RuntimeError(
            "No simulator backend found. Install qiskit-aer or ensure BasicAer is available."
        ) from e


def chunks(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_prob_class1_from_counts(counts: Dict[str, int], shots: int) -> float:
    """
    p(class=1) = probability that the LAST classical bit is 1.
    """
    if not counts:
        return float("nan")
    total = sum(counts.values())
    if total <= 0:
        total = shots

    p1 = 0
    for bitstr, c in counts.items():
        bitstr = bitstr.replace(" ", "")
        if bitstr and bitstr[-1] == "1":
            p1 += c
    return p1 / total


def run_circuits(circuits: List[Any], shots: int, batch_size: int) -> List[float]:
    backend, transpile_fn, execute_fn, kind = get_backend()
    all_probs: List[float] = []

    for batch in chunks(circuits, batch_size):
        try:
            tqc = transpile_fn(batch, backend=backend)
        except TypeError:
            tqc = transpile_fn(batch, backend)

        if kind == "aer":
            job = backend.run(tqc, shots=shots)
            result = job.result()
        else:
            assert execute_fn is not None
            job = execute_fn(tqc, backend=backend, shots=shots)
            result = job.result()

        for i in range(len(batch)):
            counts = result.get_counts(i)
            all_probs.append(extract_prob_class1_from_counts(counts, shots))

    return all_probs


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export VQC predictions by building circuits and simulating them.")
    ap.add_argument("--vqc_module", type=str, required=True, help="Python module path, e.g., src.models.vqc_hw_sim")
    ap.add_argument("--quantum_parquet", type=str, required=True, help="Parquet with patch_id and feature_col")
    ap.add_argument("--feature_col", type=str, required=True, help="Feature column, e.g., z_hw or z_sim")
    ap.add_argument("--params_npy", type=str, required=True, help="Path to best_params.npy")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    ap.add_argument("--n_qubits", type=int, required=True, help="Number of qubits (must match feature dimension)")

    # Optional: let user override. If omitted, we probe.
    ap.add_argument("--layers", type=int, default=None, help="Number of VQC layers (optional; auto-detect if omitted)")

    ap.add_argument("--shots", type=int, default=1024, help="Shots per circuit")
    ap.add_argument("--batch_size", type=int, default=64, help="How many circuits to run per batch")
    ap.add_argument("--max_rows", type=int, default=0, help="If >0, limit number of rows exported")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--probe_max_layers", type=int, default=20, help="Max layers to try when auto-detecting")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    mod = importlib.import_module(args.vqc_module)

    qpath = Path(args.quantum_parquet)
    if not qpath.exists():
        raise FileNotFoundError(f"quantum_parquet not found: {qpath}")

    ppath = Path(args.params_npy)
    if not ppath.exists():
        raise FileNotFoundError(f"params_npy not found: {ppath}")

    outpath = Path(args.out_csv)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(qpath)

    if "patch_id" not in df.columns:
        raise ValueError("Parquet must contain 'patch_id' column.")
    if args.feature_col not in df.columns:
        raise ValueError(f"Parquet missing feature_col '{args.feature_col}'. Columns: {list(df.columns)}")

    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    theta = np.load(ppath)
    theta = np.asarray(theta).reshape(-1)

    X_list = df[args.feature_col].to_list()
    if len(X_list) == 0:
        raise RuntimeError("No rows to export (empty dataframe).")

    first = np.asarray(X_list[0])
    if first.ndim != 1:
        raise ValueError(f"Feature vectors must be 1D, got shape {first.shape}")
    if len(first) != args.n_qubits:
        raise ValueError(
            f"Feature dim mismatch: '{args.feature_col}' has length {len(first)} but --n_qubits={args.n_qubits}"
        )

    # Determine layers
    if args.layers is None:
        n_layers = infer_n_layers_by_probe(
            mod,
            n_qubits=args.n_qubits,
            params=theta,
            max_layers=args.probe_max_layers,
        )
        print(f"Auto-detected n_layers={n_layers} by probing (len(params)={len(theta)})")
    else:
        n_layers = args.layers
        # Validate quickly by probing exactly that layer count
        _ = build_circuit(mod, n_qubits=args.n_qubits, n_layers=n_layers, x_angles=np.zeros(args.n_qubits), params=theta)
        print(f"Using user-specified n_layers={n_layers} (validated)")

    # Build circuits
    circuits: List[Any] = []
    patch_ids = df["patch_id"].tolist()

    for xx in X_list:
        xx = np.asarray(xx, dtype=float).reshape(-1)
        qc = build_circuit(mod, n_qubits=args.n_qubits, n_layers=n_layers, x_angles=xx, params=theta)
        circuits.append(qc)

    # Run
    probs = run_circuits(circuits, shots=args.shots, batch_size=args.batch_size)
    if len(probs) != len(patch_ids):
        raise RuntimeError(f"Internal error: probs len {len(probs)} != patch_ids len {len(patch_ids)}")

    out_df = pd.DataFrame({"patch_id": patch_ids, "p_vqc": probs})
    out_df.to_csv(outpath, index=False)

    print(f"✅ Wrote: {outpath} (rows={len(out_df)}, shots={args.shots}, batch_size={args.batch_size})")


if __name__ == "__main__":
    main()