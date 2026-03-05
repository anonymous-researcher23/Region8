import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# -------------------- VQC helpers --------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def make_vqc_circuit(n_layers: int, angles: np.ndarray, w_params: np.ndarray) -> QuantumCircuit:
    n_qubits = len(angles)
    per_layer = 2 * n_qubits
    if len(w_params) != n_layers * per_layer:
        raise ValueError(f"w params length mismatch: got {len(w_params)} expected {n_layers*per_layer}")

    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        # data embedding
        for q in range(n_qubits):
            qc.ry(float(angles[q]), q)

        # trainable
        for q in range(n_qubits):
            qc.ry(float(w_params[idx + q]), q)
        idx += n_qubits
        for q in range(n_qubits):
            qc.rz(float(w_params[idx + q]), q)
        idx += n_qubits

        # ring entangle
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)

    qc.measure_all()
    return qc

def vqc_prob_one(angles: np.ndarray, best_params_npy: str, n_layers: int,
                 shots: int, seed: int, backend: AerSimulator) -> float:
    params = np.load(best_params_npy).astype(np.float32)
    w = params[:-2]
    alpha = float(params[-2])
    beta = float(params[-1])

    qc = make_vqc_circuit(n_layers, angles, w)
    tqc = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=seed)
    job = backend.run(tqc, shots=shots, seed_simulator=seed)
    res = job.result()
    counts = res.get_counts()

    total = sum(counts.values())
    ones = sum(v for k, v in counts.items() if k[-1] == "1")  # qubit0
    p_meas1 = ones / total if total else 0.0
    zexp = (1 - p_meas1) - p_meas1  # p0 - p1
    return float(sigmoid(alpha * zexp + beta))

def vqc_batch_probs(df: pd.DataFrame, angle_col: str, best_params: str, n_layers: int,
                    shots: int, seed: int) -> np.ndarray:
    backend = AerSimulator(method="automatic")
    out = np.zeros(len(df), dtype=np.float32)
    for i, angles in enumerate(df[angle_col].to_numpy()):
        out[i] = vqc_prob_one(np.asarray(angles, dtype=np.float32), best_params, n_layers, shots, seed, backend)
    return out


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantum_test", required=True, help="test_quantum.parquet")
    ap.add_argument("--patch_test", required=True, help="test.parquet (patches) for y/flood_frac")
    ap.add_argument("--logreg_model", required=True)
    ap.add_argument("--svm_model", required=True)

    ap.add_argument("--best_params_sim", required=True, help="outputs/vqc_sim/.../best_params.npy")
    ap.add_argument("--best_params_hw_sim", required=True, help="outputs/vqc_hw_sim/.../best_params.npy")
    ap.add_argument("--n_layers_sim", type=int, default=2)
    ap.add_argument("--n_layers_hw_sim", type=int, default=2)
    ap.add_argument("--shots_sim", type=int, default=256)
    ap.add_argument("--shots_hw_sim", type=int, default=256)

    ap.add_argument("--hw_run", required=True, help="outputs/vqc_hardware/run_xxx (must have predictions.csv)")
    ap.add_argument("--out_dir", default="outputs/model_compare")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    keys = ["scene_id","patch_id","row","col"]

    q = pd.read_parquet(args.quantum_test)
    ptest = pd.read_parquet(args.patch_test)[keys + ["y","flood_frac"]]
    df = ptest.merge(q[keys + ["z_hw","theta_sim","theta_hw"]], on=keys, how="left")

    # hardware predictions (subset)
    hw = pd.read_csv(os.path.join(args.hw_run, "predictions.csv"))[keys + ["p1"]].rename(columns={"p1":"p_vqc_hw"})
    df = df.merge(hw, on=keys, how="inner")  # INNER: only points hardware actually evaluated

    # Labels
    y = df["y"].astype(int).to_numpy()

    # Classical probs
    Z = np.vstack(df["z_hw"].to_numpy())  # (N,8)
    logreg = joblib.load(args.logreg_model)
    svm = joblib.load(args.svm_model)
    p_logreg = logreg.predict_proba(Z)[:,1]
    p_svm = svm.predict_proba(Z)[:,1]

    # VQC sim probs
    p_vqc_sim = vqc_batch_probs(df, "theta_sim", args.best_params_sim, args.n_layers_sim, args.shots_sim, args.seed)
    p_vqc_hw_sim = vqc_batch_probs(df, "theta_hw", args.best_params_hw_sim, args.n_layers_hw_sim, args.shots_hw_sim, args.seed)

    p_vqc_hw = df["p_vqc_hw"].to_numpy(dtype=np.float32)

    probs = {
        "LogReg": p_logreg,
        "SVM-RBF": p_svm,
        "VQC-sim (16D)": p_vqc_sim,
        "VQC-hw-sim (8D)": p_vqc_hw_sim,
        "VQC-hw (real)": p_vqc_hw,
    }

    # -------- ROC plot
    plt.figure()
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (aligned to hardware subset, N={len(y)})")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_all_models.png"), dpi=200)
    plt.close()

    # -------- PR plot
    plt.figure()
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        apv = average_precision_score(y, p)
        plt.plot(rec, prec, label=f"{name} (AP={apv:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curves (aligned to hardware subset, N={len(y)})")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pr_all_models.png"), dpi=200)
    plt.close()

    # -------- Confusions @thr
    out = {"N_aligned_to_hw": int(len(y)), "thr": float(args.thr), "models": {}}
    for name, p in probs.items():
        pred = (p >= args.thr).astype(int)
        cm = confusion_matrix(y, pred).tolist()
        out["models"][name] = {"cm": cm}
    with open(os.path.join(args.out_dir, "confusions.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Save the aligned per-sample probabilities too (super useful)
    out_df = df[keys + ["y","flood_frac"]].copy()
    out_df["p_logreg"] = p_logreg
    out_df["p_svm"] = p_svm
    out_df["p_vqc_sim"] = p_vqc_sim
    out_df["p_vqc_hw_sim"] = p_vqc_hw_sim
    out_df["p_vqc_hw"] = p_vqc_hw
    out_df.to_csv(os.path.join(args.out_dir, "aligned_probs.csv"), index=False)

    print("Saved:", args.out_dir)
    print("N used (hardware-aligned):", len(y))

if __name__ == "__main__":
    main()
