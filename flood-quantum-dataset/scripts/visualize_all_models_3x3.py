import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# -------------------- robust column picking --------------------

def pick_col(df: pd.DataFrame, base: str) -> str:
    """
    Pick a column name even after pandas merges create suffixes (_x/_y).
    Priority:
      1) base
      2) base_y (usually from later merge)
      3) base_x
    """
    if base in df.columns:
        return base
    if f"{base}_y" in df.columns:
        return f"{base}_y"
    if f"{base}_x" in df.columns:
        return f"{base}_x"
    # last resort: any col that starts with base
    for c in df.columns:
        if c.startswith(base):
            return c
    raise KeyError(f"Could not find column for '{base}'. Available: {df.columns.tolist()}")


def pick_flood_frac_column(df: pd.DataFrame) -> str | None:
    for c in ["flood_frac", "flood_frac_y", "flood_frac_x"]:
        if c in df.columns:
            return c
    return None


# -------------------- patch helpers --------------------

def load_patch(x_path: str) -> np.ndarray:
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"x_path not found: {x_path}")

    if x_path.endswith(".npy"):
        x = np.load(x_path)
    elif x_path.endswith(".npz"):
        z = np.load(x_path)
        x = z["x"] if "x" in z else z[z.files[0]]
    else:
        raise ValueError(f"Unsupported patch file type: {x_path}")

    return np.asarray(x, dtype=np.float32)


def to_chw(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {x.shape}")
    # if (H,W,C) -> (C,H,W)
    if x.shape[-1] <= 16 and x.shape[0] >= 16 and x.shape[1] >= 16:
        x = np.transpose(x, (2, 0, 1))
    return x


def norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        return np.zeros_like(a)
    a = (a - lo) / (hi - lo)
    return np.clip(a, 0, 1)


# -------------------- VQC sim (same logic as your training script) --------------------

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
        # data embed
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


def vqc_prob_one(angles: np.ndarray, best_params_npy: str, n_layers: int, shots: int, seed: int) -> float:
    """
    Returns p(flood) using:
      measure qubit0 -> p_meas1
      zexp = p0 - p1
      p = sigmoid(alpha*zexp + beta)
    """
    params = np.load(best_params_npy).astype(np.float32)
    w = params[:-2]
    alpha = float(params[-2])
    beta = float(params[-1])

    qc = make_vqc_circuit(n_layers, angles, w)

    backend = AerSimulator(method="automatic")
    tqc = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=seed)
    job = backend.run(tqc, shots=shots, seed_simulator=seed)
    res = job.result()
    counts = res.get_counts()

    total = sum(counts.values())
    ones = sum(v for k, v in counts.items() if k[-1] == "1")  # qubit0
    p_meas1 = ones / total if total else 0.0

    zexp = (1 - p_meas1) - p_meas1  # p0 - p1
    return float(sigmoid(alpha * zexp + beta))


# -------------------- panels --------------------

def panel_model(ax, name: str, p: float | None, thr: float):
    ax.axis("off")
    if p is None or (isinstance(p, float) and np.isnan(p)):
        ax.text(0.5, 0.55, f"{name}\nN/A", ha="center", va="center", fontsize=14, fontweight="bold")
        return

    pred = 1 if p >= thr else 0
    ax.text(0.5, 0.70, name, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.48, f"P(flood)={p:.3f}", ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.28, f"Pred={pred}", ha="center", va="center", fontsize=12)

    # probability bar
    ax.add_patch(plt.Rectangle((0.12, 0.12), 0.76, 0.06, fill=False, lw=1))
    ax.add_patch(plt.Rectangle((0.12, 0.12), 0.76 * max(0.0, min(1.0, p)), 0.06, fill=True))
    ax.text(0.12, 0.02, "0", fontsize=10)
    ax.text(0.86, 0.02, "1", fontsize=10)
    ax.text(0.50, 0.02, f"thr={thr:.2f}", fontsize=10, ha="center")


def panel_gt(ax, y_true: int, flood_frac: float):
    ax.axis("off")
    lab = "FLOOD" if y_true == 1 else "NON-FLOOD"
    ax.text(0.5, 0.65, "Ground Truth", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, lab, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.22, f"flood_frac={flood_frac:.3f}", ha="center", va="center", fontsize=12)


def save_3x3(out_path, x_chw, meta, thr,
             p_logreg, p_svm, p_vqc_sim, p_vqc_hw_sim, p_vqc_hw):
    # channels: [VV,VH,B2,B3,B4,B8]
    vv = x_chw[0]; vh = x_chw[1]
    b2 = x_chw[2]; b3 = x_chw[3]; b4 = x_chw[4]
    rgb = norm01(np.stack([b4, b3, b2], axis=-1))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])

    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])

    ax00.imshow(rgb); ax00.set_title("Sentinel-2 RGB (B4,B3,B2)"); ax00.axis("off")
    ax01.imshow(norm01(vv), cmap="gray"); ax01.set_title("SAR VV"); ax01.axis("off")
    ax02.imshow(norm01(vh), cmap="gray"); ax02.set_title("SAR VH"); ax02.axis("off")

    panel_gt(ax10, meta["y_true"], meta["flood_frac"])
    panel_model(ax11, "LogReg", p_logreg, thr)
    panel_model(ax12, "SVM-RBF", p_svm, thr)

    panel_model(ax20, "VQC-sim (16D)", p_vqc_sim, thr)
    panel_model(ax21, "VQC-hw-sim (8D)", p_vqc_hw_sim, thr)
    panel_model(ax22, "VQC-hw (real)", p_vqc_hw, thr)

    title = (
        f"{meta['cat']} | scene={meta['scene_id']} patch={meta['patch_id']} (r{meta['row']},c{meta['col']}) "
        f"| y_true={meta['y_true']} flood_frac={meta['flood_frac']:.3f}"
    )
    fig.suptitle(title, fontsize=12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples_csv", required=True)
    ap.add_argument("--patch_parquet", required=True)
    ap.add_argument("--quantum_parquet", required=True)
    ap.add_argument("--hw_predictions", required=True)

    ap.add_argument("--logreg_model", required=True)
    ap.add_argument("--svm_model", required=True)

    ap.add_argument("--best_params_sim", required=True, help="best_params.npy for 16D/theta_sim")
    ap.add_argument("--best_params_hw_sim", required=True, help="best_params.npy for 8D/theta_hw")

    ap.add_argument("--out_dir", default="outputs/all_models_3x3")
    ap.add_argument("--n_layers_sim", type=int, default=2)
    ap.add_argument("--n_layers_hw_sim", type=int, default=2)
    ap.add_argument("--shots_sim", type=int, default=256)
    ap.add_argument("--shots_hw_sim", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max_per_cat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ex = pd.read_csv(args.examples_csv)

    # normalize flood_frac column in examples
    ff_col = pick_flood_frac_column(ex)
    ex["flood_frac_use"] = ex[ff_col] if ff_col else np.nan

    patches = pd.read_parquet(args.patch_parquet)[["scene_id","patch_id","row","col","x_path"]]

    # quantum parquet has the clean features; we will suffix to avoid collisions
    qdf = pd.read_parquet(args.quantum_parquet)[
        ["scene_id","patch_id","row","col","z_hw","theta_hw","theta_sim"]
    ].rename(columns={
        "z_hw": "z_hw_q",
        "theta_hw": "theta_hw_q",
        "theta_sim": "theta_sim_q",
    })

    # hardware predictions
    hw = pd.read_csv(args.hw_predictions)[["scene_id","patch_id","row","col","p1"]].rename(columns={"p1":"p_vqc_hw"})

    keys = ["scene_id","patch_id","row","col"]
    df = ex.merge(patches, on=keys, how="left").merge(qdf, on=keys, how="left").merge(hw, on=keys, how="left")

    if df["x_path"].isna().any():
        miss = df[df["x_path"].isna()].head(10)
        raise RuntimeError(f"Some rows missing x_path after merge. Sample:\n{miss}")

    # limit per category
    keep = []
    for cat in ["TP","FP","FN","TN"]:
        keep.append(df[df["cat"] == cat].head(args.max_per_cat))
    df = pd.concat(keep, ignore_index=True)

    logreg = joblib.load(args.logreg_model)
    svm = joblib.load(args.svm_model)

    print("Saving 3x3 figures:", len(df), "->", args.out_dir)

    for i, r in df.iterrows():
        x = to_chw(load_patch(r["x_path"]))

        # use the clean quantum parquet features we renamed
        z = np.asarray(r["z_hw_q"], dtype=np.float32).reshape(1, -1)
        theta_sim = np.asarray(r["theta_sim_q"], dtype=np.float32)
        theta_hw = np.asarray(r["theta_hw_q"], dtype=np.float32)

        p_logreg = float(logreg.predict_proba(z)[0, 1])
        p_svm = float(svm.predict_proba(z)[0, 1])

        p_vqc_sim = vqc_prob_one(theta_sim, args.best_params_sim, args.n_layers_sim, args.shots_sim, args.seed)
        p_vqc_hw_sim = vqc_prob_one(theta_hw, args.best_params_hw_sim, args.n_layers_hw_sim, args.shots_hw_sim, args.seed)

        p_vqc_hw = float(r["p_vqc_hw"]) if not pd.isna(r["p_vqc_hw"]) else None

        meta = {
            "cat": str(r["cat"]),
            "scene_id": str(r["scene_id"]),
            "patch_id": str(r["patch_id"]),
            "row": int(r["row"]),
            "col": int(r["col"]),
            "y_true": int(r["y_true"]),
            "flood_frac": float(r["flood_frac_use"]) if not pd.isna(r["flood_frac_use"]) else float("nan"),
        }

        out_path = os.path.join(
            args.out_dir, meta["cat"],
            f"{i:04d}_{meta['scene_id']}_{meta['row']}_{meta['col']}.png"
        )

        save_3x3(out_path, x, meta, args.thr,
                 p_logreg, p_svm, p_vqc_sim, p_vqc_hw_sim, p_vqc_hw)

    print("Done. Open:", args.out_dir, "TP/FP/FN/TN folders.")


if __name__ == "__main__":
    main()
