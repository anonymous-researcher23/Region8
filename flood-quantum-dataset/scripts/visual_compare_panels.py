#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# -------------------------
# Helpers
# -------------------------
def _load_patch_array(x_path: str) -> np.ndarray:
    """
    Load patch tensor from x_path. Supports .npy and .npz (with 'x' key).
    Expected shape: (H, W, C) or (C, H, W).
    """
    p = Path(x_path)
    if not p.exists():
        raise FileNotFoundError(f"x_path not found: {x_path}")

    if p.suffix.lower() == ".npy":
        x = np.load(p)
    elif p.suffix.lower() == ".npz":
        z = np.load(p)
        if "x" in z:
            x = z["x"]
        else:
            # try first array
            x = z[list(z.keys())[0]]
    else:
        raise ValueError(f"Unsupported x_path format: {p.suffix} (use .npy or .npz)")

    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D patch array, got shape {x.shape} for {x_path}")

    # Make (H,W,C)
    if x.shape[0] in (2, 4, 6, 8, 10, 12) and x.shape[2] not in (2, 4, 6, 8, 10, 12):
        # likely (C,H,W)
        x = np.transpose(x, (1, 2, 0))

    return x


def _normalize_for_display(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize to [0,1] per-image for visualization only.
    """
    a = img.astype(np.float32)
    lo = np.nanpercentile(a, 2)
    hi = np.nanpercentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + eps), 0.0, 1.0)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return a


def _make_rgb_from_bands(x: np.ndarray, band_map: Dict[str, int]) -> np.ndarray:
    """
    Create RGB from B4,B3,B2 if present.
    """
    for k in ["B4", "B3", "B2"]:
        if k not in band_map:
            raise ValueError(f"band_map missing {k}. Current band_map keys={list(band_map.keys())}")

    r = x[:, :, band_map["B4"]]
    g = x[:, :, band_map["B3"]]
    b = x[:, :, band_map["B2"]]
    rgb = np.stack([r, g, b], axis=-1)
    return _normalize_for_display(rgb)


def _load_parquet_indexed(path: Path, key: str = "patch_id") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if key not in df.columns:
        raise ValueError(f"Expected '{key}' column in {path}, got {df.columns.tolist()}")
    return df.set_index(key, drop=False)


def _predict_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # map to [0,1] with sigmoid (rough)
        s = np.clip(s, -40, 40)
        return 1.0 / (1.0 + np.exp(-s))
    raise ValueError("Model must support predict_proba or decision_function.")


def _draw_prob_bar(ax, p: float, thr: float = 0.5):
    """
    Draw a horizontal probability bar [0..1] with threshold marker.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # bar background
    ax.add_patch(plt.Rectangle((0, 0.35), 1.0, 0.18, fill=False, linewidth=1))

    # filled portion
    ax.add_patch(plt.Rectangle((0, 0.35), float(p), 0.18, linewidth=0))

    # threshold marker
    ax.plot([thr, thr], [0.30, 0.60], linewidth=1)

    ax.text(0.0, 0.10, "0", ha="left", va="center", fontsize=9)
    ax.text(1.0, 0.10, "1", ha="right", va="center", fontsize=9)
    ax.text(thr, 0.10, f"thr={thr:.2f}", ha="center", va="center", fontsize=9)


def _confusion_tag(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


# -------------------------
# Main plotting
# -------------------------
def render_panel(
    out_path: Path,
    scene: str,
    patch_id: str,
    row: int,
    col: int,
    y_true: int,
    flood_frac: float,
    x: np.ndarray,
    band_map: Dict[str, int],
    preds: Dict[str, float],
    thr: float = 0.5,
):
    """
    Render one panel similar to your examples.
    preds keys recommended:
      LogReg, SVM-RBF, VQC-sim (16D), VQC-hw-sim (8D), VQC-hw (real)
    """
    # Decide confusion label based on one reference model (optional)
    # We'll use SVM-RBF if present else LogReg else first key
    ref_key = "SVM-RBF" if "SVM-RBF" in preds else ("LogReg" if "LogReg" in preds else list(preds.keys())[0])
    y_pred_ref = int(preds[ref_key] >= thr)
    tag = _confusion_tag(y_true, y_pred_ref)

    title = (
        f"{tag} | scene={scene} patch={patch_id} (r{row},c{col}) | "
        f"y_true={y_true} flood_frac={flood_frac:.3f}"
    )

    # Images
    rgb = _make_rgb_from_bands(x, band_map)
    vv = _normalize_for_display(x[:, :, band_map["VV"]]) if "VV" in band_map else None
    vh = _normalize_for_display(x[:, :, band_map["VH"]]) if "VH" in band_map else None

    # Layout
    fig = plt.figure(figsize=(14, 8), dpi=150)
    fig.suptitle(title, fontsize=12, y=0.98)

    # Top row: images
    ax1 = fig.add_axes([0.06, 0.70, 0.26, 0.22])
    ax2 = fig.add_axes([0.37, 0.70, 0.26, 0.22])
    ax3 = fig.add_axes([0.68, 0.70, 0.26, 0.22])

    ax1.imshow(rgb)
    ax1.set_title("Sentinel-2 RGB (B4,B3,B2)", fontsize=10)
    ax1.axis("off")

    if vv is not None:
        ax2.imshow(vv, cmap="gray", vmin=0, vmax=1)
        ax2.set_title("SAR VV", fontsize=10)
        ax2.axis("off")
    else:
        ax2.axis("off")
        ax2.set_title("SAR VV (missing)", fontsize=10)

    if vh is not None:
        ax3.imshow(vh, cmap="gray", vmin=0, vmax=1)
        ax3.set_title("SAR VH", fontsize=10)
        ax3.axis("off")
    else:
        ax3.axis("off")
        ax3.set_title("SAR VH (missing)", fontsize=10)

    # Middle: Ground truth + 2 models
    gt_ax = fig.add_axes([0.06, 0.38, 0.26, 0.24])
    gt_ax.axis("off")
    gt_ax.text(0.5, 0.75, "Ground Truth", ha="center", va="center", fontsize=12, fontweight="bold")
    gt_ax.text(
        0.5, 0.48, "FLOOD" if y_true == 1 else "NON-FLOOD",
        ha="center", va="center", fontsize=14, fontweight="bold"
    )
    gt_ax.text(0.5, 0.25, f"flood_frac={flood_frac:.3f}", ha="center", va="center", fontsize=10)

    # Two top-model panels (LogReg / SVM)
    top_models = [k for k in ["LogReg", "SVM-RBF"] if k in preds]
    # fallback: take first two keys
    if len(top_models) < 2:
        top_models = list(preds.keys())[:2]

    def draw_model_block(x0, name):
        axT = fig.add_axes([x0, 0.40, 0.26, 0.20])
        axT.axis("off")
        p = float(preds[name])
        axT.text(0.5, 0.78, name, ha="center", va="center", fontsize=12, fontweight="bold")
        axT.text(0.5, 0.52, f"P(flood)={p:.3f}", ha="center", va="center", fontsize=10)
        axT.text(0.5, 0.30, f"Pred={int(p >= thr)}", ha="center", va="center", fontsize=10)
        bar_ax = fig.add_axes([x0 + 0.03, 0.38, 0.20, 0.06])
        _draw_prob_bar(bar_ax, p, thr=thr)

    draw_model_block(0.37, top_models[0])
    draw_model_block(0.68, top_models[1])

    # Bottom row: up to 3 quantum models
    bottom_models = [k for k in ["VQC-sim (16D)", "VQC-hw-sim (8D)", "VQC-hw (real)"] if k in preds]
    # fallback: take remaining keys
    if len(bottom_models) == 0:
        remaining = [k for k in preds.keys() if k not in top_models]
        bottom_models = remaining[:3]

    xs = [0.06, 0.37, 0.68]
    for i, name in enumerate(bottom_models[:3]):
        x0 = xs[i]
        axB = fig.add_axes([x0, 0.07, 0.26, 0.20])
        axB.axis("off")
        p = float(preds[name])
        axB.text(0.5, 0.78, name, ha="center", va="center", fontsize=12, fontweight="bold")
        axB.text(0.5, 0.52, f"P(flood)={p:.3f}", ha="center", va="center", fontsize=10)
        axB.text(0.5, 0.30, f"Pred={int(p >= thr)}", ha="center", va="center", fontsize=10)
        bar_ax = fig.add_axes([x0 + 0.03, 0.05, 0.20, 0.06])
        _draw_prob_bar(bar_ax, p, thr=thr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser("Generate visual comparison panels for selected patches.")

    ap.add_argument("--patch_parquet", type=str, required=True,
                    help="Patch parquet with x_path, y, flood_frac, scene_id/patch_id/row/col.")
    ap.add_argument("--quantum_parquet", type=str, required=False,
                    help="Quantum parquet with features used by classical models (optional).")

    ap.add_argument("--logreg_model", type=str, required=False, help="joblib for LogReg model")
    ap.add_argument("--svm_model", type=str, required=False, help="joblib for SVM-RBF model")
    ap.add_argument("--quantum_feature_col", type=str, default="z_hw",
                    help="Feature vector column inside quantum_parquet for classical models (e.g., z_hw or theta_region8)")

    ap.add_argument("--preds_csv", type=str, default=None,
                    help=("Optional CSV/Parquet with per-patch probabilities for any models "
                          "(must include patch_id and columns like p_vqc_sim16, p_vqc_hw_sim8, p_vqc_hw)."))

    ap.add_argument("--out_dir", type=str, default="outputs/visual_panels")
    ap.add_argument("--thr", type=float, default=0.5)

    ap.add_argument("--n_each", type=int, default=2,
                    help="How many examples to auto-pick for each TP/TN/FP/FN category (based on SVM if available).")
    ap.add_argument("--seed", type=int, default=0)

    # Band map assumes your typical order. Change if needed.
    ap.add_argument("--band_map", type=str, default="VV:0,VH:1,B2:2,B3:3,B4:4,B8:5",
                    help="Mapping from band name to channel index in x (H,W,C).")

    return ap.parse_args()


def _parse_band_map(s: str) -> Dict[str, int]:
    out = {}
    for part in s.split(","):
        k, v = part.split(":")
        out[k.strip()] = int(v.strip())
    return out


def main():
    args = parse_args()
    np.random.seed(args.seed)

    patch_df = _load_parquet_indexed(Path(args.patch_parquet), key="patch_id")

    qdf = None
    if args.quantum_parquet:
        qdf = _load_parquet_indexed(Path(args.quantum_parquet), key="patch_id")

    preds_df = None
    if args.preds_csv:
        p = Path(args.preds_csv)
        if p.suffix.lower() in [".parquet"]:
            preds_df = pd.read_parquet(p).set_index("patch_id", drop=False)
        else:
            preds_df = pd.read_csv(p).set_index("patch_id", drop=False)

    logreg = joblib.load(args.logreg_model) if args.logreg_model else None
    svm = joblib.load(args.svm_model) if args.svm_model else None

    band_map = _parse_band_map(args.band_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If we can compute classical probabilities, do it once for all rows
    p_lr = None
    p_svm = None
    if qdf is not None and (logreg is not None or svm is not None):
        X = np.vstack([np.asarray(v, dtype=np.float32) for v in qdf[args.quantum_feature_col].values])
        if logreg is not None:
            p_lr = _predict_proba_sklearn(logreg, X)
        if svm is not None:
            p_svm = _predict_proba_sklearn(svm, X)

        # attach to qdf order
        qdf = qdf.copy()
        if p_lr is not None:
            qdf["p_logreg"] = p_lr
        if p_svm is not None:
            qdf["p_svm"] = p_svm

    # Choose a reference probability for auto-picking TP/TN/FP/FN
    # Priority: SVM > LogReg > any preds_df column
    ref_col = None
    ref_source = None
    if qdf is not None and "p_svm" in qdf.columns:
        ref_col, ref_source = "p_svm", "qdf"
    elif qdf is not None and "p_logreg" in qdf.columns:
        ref_col, ref_source = "p_logreg", "qdf"
    elif preds_df is not None:
        # pick first probability-like column
        cand = [c for c in preds_df.columns if c.startswith("p_")]
        if cand:
            ref_col, ref_source = cand[0], "preds"

    if ref_col is None:
        raise RuntimeError(
            "Cannot auto-pick TP/TN/FP/FN: provide --svm_model + --quantum_parquet, "
            "or provide --preds_csv with p_* columns."
        )

    # Build a merged index list for picking
    idx = patch_df.index.intersection(qdf.index) if qdf is not None else patch_df.index
    base = patch_df.loc[idx].copy()

    if ref_source == "qdf":
        base["p_ref"] = qdf.loc[idx, ref_col].values
    else:
        base["p_ref"] = preds_df.loc[idx, ref_col].values

    y = base["y"].astype(int).values
    p = base["p_ref"].astype(float).values
    yhat = (p >= args.thr).astype(int)

    tags = np.array([_confusion_tag(int(yt), int(yp)) for yt, yp in zip(y, yhat)])

    selected_patch_ids: List[str] = []
    for t in ["TP", "TN", "FP", "FN"]:
        cand = base.index[tags == t].tolist()
        if not cand:
            continue
        # pick those with most confident p (for readability)
        cand_sorted = sorted(
            cand,
            key=lambda pid: abs(float(base.loc[pid, "p_ref"]) - args.thr),
            reverse=True
        )
        selected_patch_ids += cand_sorted[: args.n_each]

    # Render panels
    for pid in selected_patch_ids:
        row = patch_df.loc[pid]
        x = _load_patch_array(row["x_path"])

        preds: Dict[str, float] = {}

        # Classical
        if qdf is not None:
            if "p_logreg" in qdf.columns:
                preds["LogReg"] = float(qdf.loc[pid, "p_logreg"])
            if "p_svm" in qdf.columns:
                preds["SVM-RBF"] = float(qdf.loc[pid, "p_svm"])

        # Optional quantum/hardware preds from file
        if preds_df is not None and pid in preds_df.index:
            # map common naming to nicer labels
            mapping = {
                "p_vqc_sim16": "VQC-sim (16D)",
                "p_vqc_hw_sim8": "VQC-hw-sim (8D)",
                "p_vqc_hw": "VQC-hw (real)",
            }
            for c, nice in mapping.items():
                if c in preds_df.columns:
                    preds[nice] = float(preds_df.loc[pid, c])

        scene = row.get("scene_id", "unknown_scene")
        out_path = out_dir / f"{pid}.png"

        render_panel(
            out_path=out_path,
            scene=str(scene),
            patch_id=str(pid),
            row=int(row.get("row", -1)),
            col=int(row.get("col", -1)),
            y_true=int(row["y"]),
            flood_frac=float(row.get("flood_frac", np.nan)),
            x=x,
            band_map=band_map,
            preds=preds,
            thr=float(args.thr),
        )

        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()