#!/usr/bin/env python3
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
import joblib


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_quantum", required=True, help="test_quantum.parquet (must include patch_id,y)")
    ap.add_argument("--pca_feature_col", default="z_hw", help="PCA feature column for classical models (z_hw or z_sim)")
    ap.add_argument("--region8_feature_col", default="region8_z", help="Region8 feature column for classical models")
    ap.add_argument("--logreg_model_pca", required=True, help="joblib LogReg model trained on PCA features")
    ap.add_argument("--svm_model_pca", required=True, help="joblib SVM model trained on PCA features")
    ap.add_argument("--logreg_model_region8", required=True, help="joblib LogReg model trained on region8 features")
    ap.add_argument("--svm_model_region8", required=True, help="joblib SVM model trained on region8 features")

    ap.add_argument("--vqc_preds_pca_csv", default="", help="CSV with patch_id,p_vqc for PCA-VQC (optional)")
    ap.add_argument("--vqc_preds_region8_csv", required=True, help="CSV with patch_id,p_vqc for Region8-VQC")

    ap.add_argument("--out_dir", default="outputs/panel_compare", help="Output directory")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for acc/f1")
    return ap.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def get_proba_from_model(model, X):
    # Works for LogReg and SVC(probability=True). If SVC has no proba, use decision_function -> sigmoid-ish ranking.
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        return p
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # convert to 0-1 range for plotting; AUC is fine with raw scores too, but we keep it consistent
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    raise RuntimeError("Model has neither predict_proba nor decision_function")


def compute_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "acc": float(accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat)),
    }


def plot_roc(ax, y, p, title):
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    test = pd.read_parquet(args.test_quantum)[["patch_id", "y", args.pca_feature_col, args.region8_feature_col]].copy()

    # Expand feature vectors into 2D arrays
    X_pca = np.stack(test[args.pca_feature_col].to_numpy())
    X_r8  = np.stack(test[args.region8_feature_col].to_numpy())
    y = test["y"].to_numpy().astype(int)

    # Load models
    logreg_pca = joblib.load(args.logreg_model_pca)
    svm_pca = joblib.load(args.svm_model_pca)
    logreg_r8 = joblib.load(args.logreg_model_region8)
    svm_r8 = joblib.load(args.svm_model_region8)

    # Predict
    p_logreg_pca = get_proba_from_model(logreg_pca, X_pca)
    p_svm_pca = get_proba_from_model(svm_pca, X_pca)
    p_logreg_r8 = get_proba_from_model(logreg_r8, X_r8)
    p_svm_r8 = get_proba_from_model(svm_r8, X_r8)

    # VQC preds
    # PCA-VQC optional because your exporter is broken right now; we can leave it blank.
    p_vqc_pca = None
    if args.vqc_preds_pca_csv:
        vqc_pca = pd.read_csv(args.vqc_preds_pca_csv)
        m = test[["patch_id", "y"]].merge(vqc_pca, on="patch_id", how="inner")
        p_vqc_pca = m["p_vqc"].to_numpy()

    vqc_r8 = pd.read_csv(args.vqc_preds_region8_csv)
    m2 = test[["patch_id", "y"]].merge(vqc_r8, on="patch_id", how="inner")
    p_vqc_r8 = m2["p_vqc"].to_numpy()
    y_r8_vqc = m2["y"].to_numpy().astype(int)

    # Metrics table
    rows = []
    rows.append({"rep":"PCA", "model":"LogReg", **compute_metrics(y, p_logreg_pca, args.thr)})
    rows.append({"rep":"PCA", "model":"SVM",    **compute_metrics(y, p_svm_pca, args.thr)})
    if p_vqc_pca is not None:
        rows.append({"rep":"PCA", "model":"VQC", **compute_metrics(y, p_vqc_pca, args.thr)})
    else:
        rows.append({"rep":"PCA", "model":"VQC", "auc":np.nan, "acc":np.nan, "f1":np.nan})

    rows.append({"rep":"Region8", "model":"LogReg", **compute_metrics(y, p_logreg_r8, args.thr)})
    rows.append({"rep":"Region8", "model":"SVM",    **compute_metrics(y, p_svm_r8, args.thr)})
    rows.append({"rep":"Region8", "model":"VQC",    **compute_metrics(y_r8_vqc, p_vqc_r8, args.thr)})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "panel_metrics.csv", index=False)
    print("✅ wrote", out_dir / "panel_metrics.csv")

    # Panel ROC plot: 2 rows (PCA, Region8) x 3 cols (LogReg, SVM, VQC)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    plot_roc(axes[0,0], y, p_logreg_pca, "PCA – LogReg")
    plot_roc(axes[0,1], y, p_svm_pca, "PCA – SVM")
    if p_vqc_pca is not None:
        plot_roc(axes[0,2], y, p_vqc_pca, "PCA – VQC")
    else:
        axes[0,2].set_title("PCA – VQC (missing)")
        axes[0,2].axis("off")

    plot_roc(axes[1,0], y, p_logreg_r8, "Region8 – LogReg")
    plot_roc(axes[1,1], y, p_svm_r8, "Region8 – SVM")
    plot_roc(axes[1,2], y_r8_vqc, p_vqc_r8, "Region8 – VQC")

    plt.tight_layout()
    figpath = out_dir / "panel_roc.png"
    plt.savefig(figpath, dpi=200)
    print("✅ wrote", figpath)


if __name__ == "__main__":
    main()