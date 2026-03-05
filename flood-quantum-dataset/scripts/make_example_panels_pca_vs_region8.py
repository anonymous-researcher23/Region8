#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Create qualitative example panels (like screenshot) for PCA vs Region8.")
    ap.add_argument("--patch_test_parquet", required=True, help="patch parquet with x_path (and optionally bands)")
    ap.add_argument("--quantum_pca_test", required=True, help="quantum parquet (for patch_id,y)")
    ap.add_argument("--region8_feat_test", required=True, help="region8 feature parquet (for patch_id,y)")

    ap.add_argument("--pca_logreg_csv", required=True, help="patch_id,p (PCA LogReg)")
    ap.add_argument("--pca_svm_csv", required=True, help="patch_id,p (PCA SVM)")
    ap.add_argument("--pca_vqc_csv", required=True, help="patch_id,p_vqc (PCA VQC)")

    ap.add_argument("--r8_logreg_csv", required=True, help="patch_id,p (Region8 LogReg)")
    ap.add_argument("--r8_svm_csv", required=True, help="patch_id,p (Region8 SVM)")
    ap.add_argument("--r8_vqc_csv", required=True, help="patch_id,p_vqc (Region8 VQC)")

    ap.add_argument("--out_dir", default="outputs/all_models_3x3/example_panels")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=12, help="How many examples total to write (across TP/FP/FN/TN)")
    return ap.parse_args()


def load_preds(csv_path: str, col: str = "p"):
    df = pd.read_csv(csv_path)
    if "patch_id" not in df.columns:
        raise ValueError(f"{csv_path} missing patch_id")
    if col not in df.columns:
        # allow p_vqc
        if "p_vqc" in df.columns:
            col = "p_vqc"
        else:
            raise ValueError(f"{csv_path} missing {col} or p_vqc")
    return df[["patch_id", col]].rename(columns={col: "p"})


def safe_imread(path: str):
    # Your repo likely stores ready-made RGB/VV/VH pngs in x_path.
    # If x_path points to a .npy or something else, adjust accordingly.
    import matplotlib.image as mpimg
    return mpimg.imread(path)


def draw_bar(ax, p, thr=0.5):
    ax.barh([0], [p])
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, thr, 1])
    ax.set_xticklabels(["0", f"thr={thr:.2f}", "1"])
    ax.text(0.02, 0.65, f"P(flood)={p:.3f}", transform=ax.transAxes)
    ax.axvline(thr, linestyle="--")


def make_panel(out_path: Path, title: str, rgb, vv, vh, y_true: int, flood_frac: float,
               p_logreg: float, p_svm: float, p_vqc: float, thr=0.5):
    fig = plt.figure(figsize=(13.5, 7.5))
    fig.suptitle(title, fontsize=12)

    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 0.9, 0.9])

    ax_rgb = fig.add_subplot(gs[0, 0]); ax_vv = fig.add_subplot(gs[0, 1]); ax_vh = fig.add_subplot(gs[0, 2])
    ax_rgb.imshow(rgb); ax_rgb.set_title("Sentinel-2 RGB"); ax_rgb.axis("off")
    ax_vv.imshow(vv, cmap="gray"); ax_vv.set_title("SAR VV"); ax_vv.axis("off")
    ax_vh.imshow(vh, cmap="gray"); ax_vh.set_title("SAR VH"); ax_vh.axis("off")

    ax_gt = fig.add_subplot(gs[1, 0]); ax_lr = fig.add_subplot(gs[1, 1]); ax_svm = fig.add_subplot(gs[1, 2])
    ax_vqc = fig.add_subplot(gs[2, 1])

    ax_gt.axis("off")
    ax_gt.text(0.1, 0.7, "Ground Truth", fontsize=12, weight="bold")
    ax_gt.text(0.1, 0.35, "FLOOD" if y_true == 1 else "NON-FLOOD", fontsize=16, weight="bold")
    ax_gt.text(0.1, 0.1, f"flood_frac={flood_frac:.3f}", fontsize=11)

    ax_lr.set_title("LogReg")
    draw_bar(ax_lr, p_logreg, thr)
    ax_lr.text(0.02, 0.15, f"Pred={int(p_logreg>=thr)}", transform=ax_lr.transAxes)

    ax_svm.set_title("SVM-RBF")
    draw_bar(ax_svm, p_svm, thr)
    ax_svm.text(0.02, 0.15, f"Pred={int(p_svm>=thr)}", transform=ax_svm.transAxes)

    ax_vqc.set_title("VQC")
    draw_bar(ax_vqc, p_vqc, thr)
    ax_vqc.text(0.02, 0.15, f"Pred={int(p_vqc>=thr)}", transform=ax_vqc.transAxes)

    # empty slots for symmetry
    ax_empty1 = fig.add_subplot(gs[2, 0]); ax_empty2 = fig.add_subplot(gs[2, 2])
    ax_empty1.axis("off"); ax_empty2.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patches = pd.read_parquet(args.patch_test_parquet)
    if "patch_id" not in patches.columns or "x_path" not in patches.columns:
        raise ValueError("patch parquet must include patch_id and x_path")

    # labels from PCA quantum parquet (same patch_id universe)
    q_pca = pd.read_parquet(args.quantum_pca_test)[["patch_id","y","flood_frac"]] if "flood_frac" in pd.read_parquet(args.quantum_pca_test).columns else pd.read_parquet(args.quantum_pca_test)[["patch_id","y"]]
    if "flood_frac" not in q_pca.columns:
        q_pca["flood_frac"] = np.nan

    base = patches[["patch_id","x_path"]].merge(q_pca, on="patch_id", how="inner")

    # Load predictions
    pca_lr = load_preds(args.pca_logreg_csv, col="p")
    pca_svm = load_preds(args.pca_svm_csv, col="p")
    pca_vqc = load_preds(args.pca_vqc_csv, col="p_vqc")

    r8_lr = load_preds(args.r8_logreg_csv, col="p")
    r8_svm = load_preds(args.r8_svm_csv, col="p")
    r8_vqc = load_preds(args.r8_vqc_csv, col="p_vqc")

    df = base.merge(pca_lr, on="patch_id", how="inner", suffixes=("",""))
    df = df.rename(columns={"p":"pca_logreg"})
    df = df.merge(pca_svm, on="patch_id").rename(columns={"p":"pca_svm"})
    df = df.merge(pca_vqc, on="patch_id").rename(columns={"p":"pca_vqc"})

    df = df.merge(r8_lr, on="patch_id").rename(columns={"p":"r8_logreg"})
    df = df.merge(r8_svm, on="patch_id").rename(columns={"p":"r8_svm"})
    df = df.merge(r8_vqc, on="patch_id").rename(columns={"p":"r8_vqc"})

    # Choose examples stratified by PCA SVM correctness (you can change to any model)
    thr = args.thr
    y = df["y"].to_numpy().astype(int)
    p = df["pca_svm"].to_numpy().astype(float)
    pred = (p >= thr).astype(int)

    idx_tp = np.where((y==1) & (pred==1))[0]
    idx_tn = np.where((y==0) & (pred==0))[0]
    idx_fp = np.where((y==0) & (pred==1))[0]
    idx_fn = np.where((y==1) & (pred==0))[0]

    # pick balanced set
    k = max(1, args.n // 4)
    picks = []
    for arr in [idx_tp, idx_fp, idx_fn, idx_tn]:
        if len(arr) == 0:
            continue
        picks.extend(arr[:k].tolist())
    picks = picks[:args.n]

    for j, i in enumerate(picks):
        row = df.iloc[i]
        # IMPORTANT: x_path must point to something we can render.
        # If x_path is a stacked tensor file, you’ll need to adapt this loader to your format.
        img = safe_imread(row["x_path"])

        # Heuristic split: if img is (H,W,3) assume it is RGB already and use as all 3 panels.
        # If your pipeline stores separate RGB/VV/VH files, update this accordingly.
        if img.ndim == 3 and img.shape[-1] >= 3:
            rgb = img[..., :3]
            vv = img[..., 0]
            vh = img[..., 1] if img.shape[-1] > 1 else img[..., 0]
        else:
            rgb = np.stack([img, img, img], axis=-1)
            vv = img
            vh = img

        # Two panels per patch: PCA and Region8
        title_base = f"patch_id={row['patch_id']} | y_true={int(row['y'])} | flood_frac={float(row['flood_frac']) if not pd.isna(row['flood_frac']) else -1:.3f}"

        make_panel(
            out_dir / f"{j:03d}_PCA8.png",
            "PCA8 — " + title_base,
            rgb, vv, vh, int(row["y"]), float(row["flood_frac"]) if not pd.isna(row["flood_frac"]) else 0.0,
            float(row["pca_logreg"]), float(row["pca_svm"]), float(row["pca_vqc"]),
            thr=thr
        )
        make_panel(
            out_dir / f"{j:03d}_Region8.png",
            "Region8 — " + title_base,
            rgb, vv, vh, int(row["y"]), float(row["flood_frac"]) if not pd.isna(row["flood_frac"]) else 0.0,
            float(row["r8_logreg"]), float(row["r8_svm"]), float(row["r8_vqc"]),
            thr=thr
        )

    print(f"✅ wrote example panels to {out_dir}")


if __name__ == "__main__":
    main()