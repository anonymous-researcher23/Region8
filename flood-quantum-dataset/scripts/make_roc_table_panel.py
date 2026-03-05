#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve


def parse_args():
    ap = argparse.ArgumentParser(description="ROC + metrics panel for PCA vs Region8 (LogReg/SVM/VQC).")
    ap.add_argument("--pca_train_quantum", required=True)
    ap.add_argument("--pca_test_quantum", required=True)
    ap.add_argument("--pca_feature", default="z_hw")

    ap.add_argument("--region8_train_feat", required=True)
    ap.add_argument("--region8_test_feat", required=True)
    ap.add_argument("--region8_feature", default="region8_z")

    ap.add_argument("--pca_vqc_preds", required=True, help="CSV patch_id,p_vqc for PCA-VQC")
    ap.add_argument("--region8_vqc_preds", required=True, help="CSV patch_id,p_vqc for Region8-VQC")

    ap.add_argument("--out_dir", default="outputs/all_models_3x3/roc_table_panel")
    ap.add_argument("--thr", type=float, default=0.5)
    return ap.parse_args()


def as_matrix(col):
    return np.stack([np.asarray(v, dtype=float) for v in col.to_list()])


def compute_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(
        auc=float(roc_auc_score(y, p)),
        acc=float(accuracy_score(y, yhat)),
        f1=float(f1_score(y, yhat)),
    )


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
    out_dir.mkdir(parents=True, exist_ok=True)

    # PCA classical (train on PCA train, eval on PCA test)
    pca_tr = pd.read_parquet(args.pca_train_quantum)[["patch_id", "y", args.pca_feature]]
    pca_te = pd.read_parquet(args.pca_test_quantum)[["patch_id", "y", args.pca_feature]]
    Xp_tr = as_matrix(pca_tr[args.pca_feature]); yp_tr = pca_tr["y"].to_numpy().astype(int)
    Xp_te = as_matrix(pca_te[args.pca_feature]); yp_te = pca_te["y"].to_numpy().astype(int)

    logreg_pca = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
    svm_pca = Pipeline([("sc", StandardScaler()), ("svm", SVC(C=5.0, gamma="scale", probability=True))])
    logreg_pca.fit(Xp_tr, yp_tr)
    svm_pca.fit(Xp_tr, yp_tr)
    p_logreg_pca = logreg_pca.predict_proba(Xp_te)[:, 1]
    p_svm_pca = svm_pca.predict_proba(Xp_te)[:, 1]

    # PCA VQC preds
    pca_vqc = pd.read_csv(args.pca_vqc_preds)
    m_pca = pca_te[["patch_id","y"]].merge(pca_vqc, on="patch_id", how="inner")
    yp_vqc = m_pca["y"].to_numpy().astype(int)
    p_vqc_pca = m_pca["p_vqc"].to_numpy().astype(float)

    # Region8 classical
    r8_tr = pd.read_parquet(args.region8_train_feat)[["patch_id", "y", args.region8_feature]]
    r8_te = pd.read_parquet(args.region8_test_feat)[["patch_id", "y", args.region8_feature]]
    Xr_tr = as_matrix(r8_tr[args.region8_feature]); yr_tr = r8_tr["y"].to_numpy().astype(int)
    Xr_te = as_matrix(r8_te[args.region8_feature]); yr_te = r8_te["y"].to_numpy().astype(int)

    logreg_r8 = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
    svm_r8 = Pipeline([("sc", StandardScaler()), ("svm", SVC(C=5.0, gamma="scale", probability=True))])
    logreg_r8.fit(Xr_tr, yr_tr)
    svm_r8.fit(Xr_tr, yr_tr)
    p_logreg_r8 = logreg_r8.predict_proba(Xr_te)[:, 1]
    p_svm_r8 = svm_r8.predict_proba(Xr_te)[:, 1]

    # Region8 VQC preds
    r8_vqc = pd.read_csv(args.region8_vqc_preds)
    m_r8 = r8_te[["patch_id","y"]].merge(r8_vqc, on="patch_id", how="inner")
    yr_vqc = m_r8["y"].to_numpy().astype(int)
    p_vqc_r8 = m_r8["p_vqc"].to_numpy().astype(float)

    # Metrics table
    rows = []
    rows.append({"rep":"PCA", "model":"LogReg", **compute_metrics(yp_te, p_logreg_pca, args.thr)})
    rows.append({"rep":"PCA", "model":"SVM", **compute_metrics(yp_te, p_svm_pca, args.thr)})
    rows.append({"rep":"PCA", "model":"VQC", **compute_metrics(yp_vqc, p_vqc_pca, args.thr)})

    rows.append({"rep":"Region8", "model":"LogReg", **compute_metrics(yr_te, p_logreg_r8, args.thr)})
    rows.append({"rep":"Region8", "model":"SVM", **compute_metrics(yr_te, p_svm_r8, args.thr)})
    rows.append({"rep":"Region8", "model":"VQC", **compute_metrics(yr_vqc, p_vqc_r8, args.thr)})

    dfm = pd.DataFrame(rows)
    dfm.to_csv(out_dir / "panel_metrics.csv", index=False)

    # Figure: top row = ROC panels, bottom row = metrics table
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.2])

    ax00 = fig.add_subplot(gs[0, 0]); ax01 = fig.add_subplot(gs[0, 1]); ax02 = fig.add_subplot(gs[0, 2])
    plot_roc(ax00, yp_te, p_logreg_pca, "PCA – LogReg")
    plot_roc(ax01, yp_te, p_svm_pca, "PCA – SVM")
    plot_roc(ax02, yp_vqc, p_vqc_pca, "PCA – VQC")

    ax10 = fig.add_subplot(gs[1, :])
    ax10.axis("off")
    tbl = ax10.table(
        cellText=np.round(dfm[["rep","model","auc","acc","f1"]].values, 4),
        colLabels=["rep","model","auc","acc","f1"],
        loc="center",
        cellLoc="center",
    )
    tbl.scale(1, 1.6)
    ax10.set_title("Metrics summary", pad=10)

    fig.suptitle("PCA vs Region-8 — LogReg / SVM / VQC (ROC + metrics)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = out_dir / "roc_table_panel.png"
    fig.savefig(out_png, dpi=200)
    print("✅ wrote", out_dir / "panel_metrics.csv")
    print("✅ wrote", out_png)


if __name__ == "__main__":
    main()