#!/usr/bin/env python3
"""
Generate ECCV-style *dual-encoding* qualitative figure:
Same patch shown once, but predictions shown side-by-side for:

  LEFT  = PCA8 encoding (e.g., z_hw)
  RIGHT = Region8 encoding (e.g., theta_region8)

Top: RGB + SAR VV + SAR VH (shared)
Middle/Bottom: probability bars for each model under each encoding.

Supports:
- Classical probs from joblib models (LogReg + SVM-RBF) using feature vectors in quantum parquets
- Optional VQC probabilities loaded from CSV/Parquet (per patch) for each encoding

Example usage:

python scripts/visual_compare_dual_encoding.py \
  --patch_parquet data/processed/patches/sen1floods11_handlabeled/test.parquet \
  --pca_quantum_parquet data/processed/quantum/sen1floods11_handlabeled/test_quantum.parquet \
  --pca_feature_col z_hw \
  --region_quantum_parquet data/processed/quantum_region8/sen1floods11_handlabeled/test_quantum.parquet \
  --region_feature_col theta_region8 \
  --pca_logreg_model outputs/models_subset_4k/logreg_z_hw.joblib \
  --pca_svm_model outputs/models_subset_4k/svm_rbf_z_hw.joblib \
  --region_logreg_model outputs/models_subset_4k/logreg_theta_region8.joblib \
  --region_svm_model outputs/models_subset_4k/svm_rbf_theta_region8.joblib \
  --out_dir outputs/visual_panels_dual \
  --thr 0.50 --n_each 2

If you also have VQC probs per patch:
--pca_preds_file outputs/preds_pca.csv --region_preds_file outputs/preds_region8.csv
(where each file has columns: patch_id, p_vqc)

Notes:
- patch_parquet must contain: patch_id, x_path, y, flood_frac, scene_id (optional), row, col
- x_path must point to .npy or .npz patch arrays with shape (H,W,C) or (C,H,W)
- Default band_map assumes channels: VV,VH,B2,B3,B4,B8 in that order. Change if yours differs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# -----------------------------
# IO + preprocessing
# -----------------------------
def load_patch_array(x_path: str) -> np.ndarray:
    p = Path(x_path)
    if not p.exists():
        raise FileNotFoundError(f"x_path not found: {x_path}")

    suf = p.suffix.lower()
    if suf == ".npy":
        x = np.load(p)
    elif suf == ".npz":
        z = np.load(p)
        if "x" in z:
            x = z["x"]
        else:
            x = z[list(z.keys())[0]]
    else:
        raise ValueError(f"Unsupported x_path format: {suf} (use .npy or .npz)")

    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D patch array, got {x.shape} for {x_path}")

    # Ensure (H,W,C)
    if x.shape[0] in (2, 4, 6, 8, 10, 12) and x.shape[2] not in (2, 4, 6, 8, 10, 12):
        x = np.transpose(x, (1, 2, 0))
    return x


def normalize_for_display(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = img.astype(np.float32)
    lo = np.nanpercentile(a, 2)
    hi = np.nanpercentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + eps), 0.0, 1.0)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return a


def make_rgb(x: np.ndarray, band_map: Dict[str, int]) -> np.ndarray:
    for k in ["B4", "B3", "B2"]:
        if k not in band_map:
            raise ValueError(f"band_map missing {k}. Got keys={list(band_map.keys())}")
    r = x[:, :, band_map["B4"]]
    g = x[:, :, band_map["B3"]]
    b = x[:, :, band_map["B2"]]
    return normalize_for_display(np.stack([r, g, b], axis=-1))


def load_indexed_parquet(path: Path, key: str = "patch_id") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if key not in df.columns:
        raise ValueError(f"Expected '{key}' in {path}, got columns: {df.columns.tolist()}")
    return df.set_index(key, drop=False)


def parse_band_map(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in s.split(","):
        k, v = part.split(":")
        out[k.strip()] = int(v.strip())
    return out


# -----------------------------
# Models / probabilities
# -----------------------------
def predict_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.clip(s, -40, 40)
        return 1.0 / (1.0 + np.exp(-s))
    raise ValueError("Model must support predict_proba or decision_function.")


def load_preds_file(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pred file not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "patch_id" not in df.columns:
        raise ValueError(f"Pred file must contain patch_id. Got columns: {df.columns.tolist()}")
    return df.set_index("patch_id", drop=False)


# -----------------------------
# Plotting utilities
# -----------------------------
def draw_prob_bar(ax, p: float, thr: float = 0.5):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Outline
    ax.add_patch(plt.Rectangle((0, 0.35), 1.0, 0.18, fill=False, linewidth=1))
    # Filled
    ax.add_patch(plt.Rectangle((0, 0.35), float(p), 0.18, linewidth=0))
    # Threshold marker
    ax.plot([thr, thr], [0.30, 0.60], linewidth=1)
    ax.text(0.0, 0.10, "0", ha="left", va="center", fontsize=8)
    ax.text(1.0, 0.10, "1", ha="right", va="center", fontsize=8)
    ax.text(thr, 0.10, f"thr={thr:.2f}", ha="center", va="center", fontsize=8)


def confusion_tag(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def render_dual_encoding_panel(
    out_path: Path,
    scene: str,
    patch_id: str,
    row: int,
    col: int,
    y_true: int,
    flood_frac: float,
    x: np.ndarray,
    band_map: Dict[str, int],
    preds_pca: Dict[str, float],
    preds_region: Dict[str, float],
    thr: float,
    title_ref: str = "SVM-RBF",
):
    # Use reference from PCA side (preferred) to label TP/TN/FP/FN
    p_ref = preds_pca.get(title_ref, next(iter(preds_pca.values())))
    tag = confusion_tag(y_true, int(p_ref >= thr))
    suptitle = (
        f"{tag} | scene={scene} patch={patch_id} (r{row},c{col}) | "
        f"y_true={y_true} flood_frac={flood_frac:.3f}"
    )

    rgb = make_rgb(x, band_map)
    vv = normalize_for_display(x[:, :, band_map["VV"]]) if "VV" in band_map else None
    vh = normalize_for_display(x[:, :, band_map["VH"]]) if "VH" in band_map else None

    fig = plt.figure(figsize=(15, 8), dpi=150)
    fig.suptitle(suptitle, fontsize=12, y=0.98)

    # --- Top row: imagery (shared) ---
    ax1 = fig.add_axes([0.06, 0.70, 0.26, 0.22])
    ax2 = fig.add_axes([0.37, 0.70, 0.26, 0.22])
    ax3 = fig.add_axes([0.68, 0.70, 0.26, 0.22])

    ax1.imshow(rgb)
    ax1.set_title("Sentinel-2 RGB (B4,B3,B2)", fontsize=10)
    ax1.axis("off")

    if vv is not None:
        ax2.imshow(vv, cmap="gray", vmin=0, vmax=1)
        ax2.set_title("SAR VV", fontsize=10)
    else:
        ax2.set_title("SAR VV (missing)", fontsize=10)
    ax2.axis("off")

    if vh is not None:
        ax3.imshow(vh, cmap="gray", vmin=0, vmax=1)
        ax3.set_title("SAR VH", fontsize=10)
    else:
        ax3.set_title("SAR VH (missing)", fontsize=10)
    ax3.axis("off")

    # --- Left block: Ground truth ---
    gt_ax = fig.add_axes([0.06, 0.38, 0.26, 0.24])
    gt_ax.axis("off")
    gt_ax.text(0.5, 0.75, "Ground Truth", ha="center", va="center", fontsize=12, fontweight="bold")
    gt_ax.text(
        0.5, 0.48, "FLOOD" if y_true == 1 else "NON-FLOOD",
        ha="center", va="center", fontsize=14, fontweight="bold"
    )
    gt_ax.text(0.5, 0.25, f"flood_frac={flood_frac:.3f}", ha="center", va="center", fontsize=10)

    # --- Two columns: PCA vs Region8 ---
    fig.text(0.45, 0.62, "PCA8 encoding", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.76, 0.62, "Region8 encoding", ha="center", va="center", fontsize=12, fontweight="bold")

    # Decide which models to show (same list for both sides)
    # Prefer: LogReg, SVM-RBF, VQC
    def _order_keys(d: Dict[str, float]) -> List[str]:
        want = ["LogReg", "SVM-RBF", "VQC"]
        keys = []
        for k in want:
            if k in d:
                keys.append(k)
        # fill with any others
        for k in d.keys():
            if k not in keys:
                keys.append(k)
        return keys

    keys = _order_keys(preds_pca)
    # keep at most 3 for clean figure
    keys = keys[:3]

    y_positions = [0.51, 0.30, 0.09]  # three rows
    # Left column (PCA)
    for k, y0 in zip(keys, y_positions):
        ax = fig.add_axes([0.37, y0, 0.26, 0.18])
        ax.axis("off")
        p = float(preds_pca[k])
        ax.text(0.5, 0.75, f"{k} (PCA8)", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(0.5, 0.50, f"P(flood)={p:.3f}", ha="center", va="center", fontsize=10)
        ax.text(0.5, 0.30, f"Pred={int(p >= thr)}", ha="center", va="center", fontsize=10)
        bar = fig.add_axes([0.40, y0 - 0.02, 0.20, 0.06])
        draw_prob_bar(bar, p, thr=thr)

    # Right column (Region8)
    for k, y0 in zip(keys, y_positions):
        ax = fig.add_axes([0.68, y0, 0.26, 0.18])
        ax.axis("off")
        if k not in preds_region:
            # if missing, display blank but don't crash
            ax.text(0.5, 0.55, f"{k} (Region8)\nmissing", ha="center", va="center", fontsize=10)
            continue
        p = float(preds_region[k])
        ax.text(0.5, 0.75, f"{k} (Region8)", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(0.5, 0.50, f"P(flood)={p:.3f}", ha="center", va="center", fontsize=10)
        ax.text(0.5, 0.30, f"Pred={int(p >= thr)}", ha="center", va="center", fontsize=10)
        bar = fig.add_axes([0.71, y0 - 0.02, 0.20, 0.06])
        draw_prob_bar(bar, p, thr=thr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Selection logic (TP/TN/FP/FN)
# -----------------------------
def pick_examples(
    patch_df: pd.DataFrame,
    p_ref: pd.Series,
    thr: float,
    n_each: int,
    seed: int,
) -> List[str]:
    rng = np.random.default_rng(seed)
    y = patch_df.loc[p_ref.index, "y"].astype(int).values
    p = p_ref.values.astype(float)
    yhat = (p >= thr).astype(int)
    tags = np.array([confusion_tag(int(yt), int(yp)) for yt, yp in zip(y, yhat)])

    picked: List[str] = []
    for t in ["TP", "TN", "FP", "FN"]:
        candidates = p_ref.index[tags == t].tolist()
        if not candidates:
            continue
        # Prefer more "confident" examples so bars are visually obvious
        candidates = sorted(candidates, key=lambda pid: abs(float(p_ref.loc[pid]) - thr), reverse=True)
        # If not enough, shuffle remaining
        top = candidates[:n_each]
        if len(top) < n_each and len(candidates) > len(top):
            rest = candidates[len(top):]
            rng.shuffle(rest)
            top += rest[: (n_each - len(top))]
        picked.extend(top[:n_each])
    return picked


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Generate dual-encoding qualitative comparison panels (PCA8 vs Region8).")

    ap.add_argument("--patch_parquet", type=str, required=True)

    ap.add_argument("--pca_quantum_parquet", type=str, required=True)
    ap.add_argument("--pca_feature_col", type=str, default="z_hw")

    ap.add_argument("--region_quantum_parquet", type=str, required=True)
    ap.add_argument("--region_feature_col", type=str, default="theta_region8")

    ap.add_argument("--pca_logreg_model", type=str, required=False)
    ap.add_argument("--pca_svm_model", type=str, required=False)
    ap.add_argument("--region_logreg_model", type=str, required=False)
    ap.add_argument("--region_svm_model", type=str, required=False)

    ap.add_argument("--pca_preds_file", type=str, default=None,
                    help="Optional CSV/Parquet with columns patch_id, p_vqc (PCA side).")
    ap.add_argument("--region_preds_file", type=str, default=None,
                    help="Optional CSV/Parquet with columns patch_id, p_vqc (Region8 side).")

    ap.add_argument("--out_dir", type=str, default="outputs/visual_panels_dual")
    ap.add_argument("--thr", type=float, default=0.5)

    ap.add_argument("--n_each", type=int, default=2, help="Examples per TP/TN/FP/FN (based on PCA SVM if available).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_examples", type=int, default=999999,
                    help="Hard cap on total rendered images (for quick runs).")

    ap.add_argument("--band_map", type=str, default="VV:0,VH:1,B2:2,B3:3,B4:4,B8:5",
                    help="Mapping of band name to channel index in x (H,W,C).")

    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    band_map = parse_band_map(args.band_map)

    patch_df = load_indexed_parquet(Path(args.patch_parquet), key="patch_id")

    pca_q = load_indexed_parquet(Path(args.pca_quantum_parquet), key="patch_id")
    reg_q = load_indexed_parquet(Path(args.region_quantum_parquet), key="patch_id")

    # Load sklearn models
    pca_logreg = joblib.load(args.pca_logreg_model) if args.pca_logreg_model else None
    pca_svm = joblib.load(args.pca_svm_model) if args.pca_svm_model else None
    reg_logreg = joblib.load(args.region_logreg_model) if args.region_logreg_model else None
    reg_svm = joblib.load(args.region_svm_model) if args.region_svm_model else None

    # Optional VQC probs
    pca_preds = load_preds_file(args.pca_preds_file)
    reg_preds = load_preds_file(args.region_preds_file)

    # Restrict to common patch ids
    common = patch_df.index.intersection(pca_q.index).intersection(reg_q.index)
    if len(common) == 0:
        raise RuntimeError("No common patch_id across patch_parquet and both quantum parquets.")
    patch_df = patch_df.loc[common]
    pca_q = pca_q.loc[common]
    reg_q = reg_q.loc[common]

    # Compute classical probabilities for ALL common patches (so we can pick TP/TN/FP/FN automatically)
    def vec_stack(df: pd.DataFrame, col: str) -> np.ndarray:
        if col not in df.columns:
            raise ValueError(f"Feature col '{col}' not in {df.columns.tolist()}")
        return np.vstack([np.asarray(v, dtype=np.float32) for v in df[col].values])

    X_pca = vec_stack(pca_q, args.pca_feature_col)
    X_reg = vec_stack(reg_q, args.region_feature_col)

    p_pca_lr = predict_proba_sklearn(pca_logreg, X_pca) if pca_logreg is not None else None
    p_pca_svm = predict_proba_sklearn(pca_svm, X_pca) if pca_svm is not None else None
    p_reg_lr = predict_proba_sklearn(reg_logreg, X_reg) if reg_logreg is not None else None
    p_reg_svm = predict_proba_sklearn(reg_svm, X_reg) if reg_svm is not None else None

    # Reference for picking: PCA SVM if available else PCA LogReg else error
    if p_pca_svm is not None:
        p_ref = pd.Series(p_pca_svm, index=common, name="p_ref")
    elif p_pca_lr is not None:
        p_ref = pd.Series(p_pca_lr, index=common, name="p_ref")
    else:
        raise RuntimeError("Need at least one PCA classical model (SVM or LogReg) to auto-pick examples.")

    chosen = pick_examples(patch_df, p_ref, thr=float(args.thr), n_each=int(args.n_each), seed=int(args.seed))
    chosen = chosen[: int(args.max_examples)]

    # Build lookup series for probabilities
    pca_lr_s = pd.Series(p_pca_lr, index=common) if p_pca_lr is not None else None
    pca_svm_s = pd.Series(p_pca_svm, index=common) if p_pca_svm is not None else None
    reg_lr_s = pd.Series(p_reg_lr, index=common) if p_reg_lr is not None else None
    reg_svm_s = pd.Series(p_reg_svm, index=common) if p_reg_svm is not None else None

    for pid in chosen:
        row = patch_df.loc[pid]
        x = load_patch_array(str(row["x_path"]))

        preds_pca: Dict[str, float] = {}
        preds_reg: Dict[str, float] = {}

        if pca_lr_s is not None:
            preds_pca["LogReg"] = float(pca_lr_s.loc[pid])
        if pca_svm_s is not None:
            preds_pca["SVM-RBF"] = float(pca_svm_s.loc[pid])

        if reg_lr_s is not None:
            preds_reg["LogReg"] = float(reg_lr_s.loc[pid])
        if reg_svm_s is not None:
            preds_reg["SVM-RBF"] = float(reg_svm_s.loc[pid])

        # Optional VQC per patch
        if pca_preds is not None and pid in pca_preds.index and "p_vqc" in pca_preds.columns:
            preds_pca["VQC"] = float(pca_preds.loc[pid, "p_vqc"])
        if reg_preds is not None and pid in reg_preds.index and "p_vqc" in reg_preds.columns:
            preds_reg["VQC"] = float(reg_preds.loc[pid, "p_vqc"])

        scene = str(row.get("scene_id", "unknown_scene"))
        r = int(row.get("row", -1))
        c = int(row.get("col", -1))
        y_true = int(row["y"])
        flood_frac = float(row.get("flood_frac", np.nan))

        out_path = out_dir / f"{pid}_dual.png"
        render_dual_encoding_panel(
            out_path=out_path,
            scene=scene,
            patch_id=str(pid),
            row=r,
            col=c,
            y_true=y_true,
            flood_frac=flood_frac,
            x=x,
            band_map=band_map,
            preds_pca=preds_pca,
            preds_region=preds_reg,
            thr=float(args.thr),
            title_ref="SVM-RBF",
        )
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()