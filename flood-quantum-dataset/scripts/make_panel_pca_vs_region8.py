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
    ap = argparse.ArgumentParser(description="2x3 ROC panel: PCA vs Region8 for LogReg/SVM/VQC.")
    ap.add_argument("--pca_quantum_test", required=True)
    ap.add_argument("--pca_quantum_train", required=True)
    ap.add_argument("--pca_feature", default="z_hw")

    ap.add_argument("--region8_feat_test", required=True)
    ap.add_argument("--region8_feat_train", required=True)
    ap.add_argument("--region8_feature", default="region8_z")

    ap.add_argument("--region8_vqc_preds", required=True)  # patch_id,p_vqc

    ap.add_argument("--out_dir", default="outputs/all_models_3x3/panel_pca_vs_region8")
    ap.add_argument("--thr", type=float, default=0.5)
    return ap.parse_args()


def as_matrix(col):
    return np.stack([np.asarray(v, dtype=float) for v in col.to_list()])


def metrics(y, p, thr=0.5):
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
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- PCA (train/test from quantum parquet) ----
    pca_tr = pd.read_parquet(args.pca_quantum_train)[["patch_id", "y", args.pca_feature]]
    pca_te = pd.read_parquet(args.pca_quantum_test)[["patch_id", "y", args.pca_feature]]

    Xp_tr = as_matrix(pca_tr[args.pca_feature])
    yp_tr = pca_tr["y"].to_numpy().astype(int)
    Xp_te = as_matrix(pca_te[args.pca_feature])
    yp_te = pca_te["y"].to_numpy().astype(int)

    logreg_pca = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, n_jobs=-1))])
    svm_pca = Pipeline([("sc", StandardScaler()), ("svm", SVC(C=5.0, gamma="scale", probability=True))])
    logreg_pca.fit(Xp_tr, yp_tr)
    svm_pca.fit(Xp_tr, yp_tr)

    p_logreg_pca = logreg_pca.predict_proba(Xp_te)[:, 1]
    p_svm_pca = svm_pca.predict_proba(Xp_te)[:, 1]

    # ---- Region8 classical (train/test from features parquet) ----
    r8_tr = pd.read_parquet(args.region8_feat_train)[["patch_id", "y", args.region8_feature]]
    r8_te = pd.read_parquet(args.region8_feat_test)[["patch_id", "y", args.region8_feature]]

    Xr_tr = as_matrix(r8_tr[args.region8_feature])
    yr_tr = r8_tr["y"].to_numpy().astype(int)
    Xr_te = as_matrix(r8_te[args.region8_feature])
    yr_te = r8_te["y"].to_numpy().astype(int)

    logreg_r8 = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, n_jobs=-1))])
    svm_r8 = Pipeline([("sc", StandardScaler()), ("svm", SVC(C=5.0, gamma="scale", probability=True))])
    logreg_r8.fit(Xr_tr, yr_tr)
    svm_r8.fit(Xr_tr, yr_tr)

    p_logreg_r8 = logreg_r8.predict_proba(Xr_te)[:, 1]
    p_svm_r8 = svm_r8.predict_proba(Xr_te)[:, 1]

    # ---- Region8 VQC preds merge on patch_id ----
    vqc = pd.read_csv(args.region8_vqc_preds)  # patch_id,p_vqc
    m = r8_te[["patch_id", "y"]].merge(vqc, on="patch_id", how="inner")
    y_vqc = m["y"].to_numpy().astype(int)
    p_vqc = m["p_vqc"].to_numpy().astype(float)

    # ---- Metrics table ----
    rows = []
    rows.append({"rep": "PCA", "model": "LogReg", **metrics(yp_te, p_logreg_pca, args.thr)})
    rows.append({"rep": "PCA", "model": "SVM", **metrics(yp_te, p_svm_pca, args.thr)})

    # PCA VQC not included here because your PCA-VQC params/circuit mismatch is unresolved.
    rows.append({"rep": "PCA", "model": "VQC", "auc": np.nan, "acc": np.nan, "f1": np.nan})

    rows.append({"rep": "Region8", "model": "LogReg", **metrics(yr_te, p_logreg_r8, args.thr)})
    rows.append({"rep": "Region8", "model": "SVM", **metrics(yr_te, p_svm_r8, args.thr)})
    rows.append({"rep": "Region8", "model": "VQC", **metrics(y_vqc, p_vqc, args.thr)})

    dfm = pd.DataFrame(rows)
    dfm.to_csv(out_dir / "panel_metrics.csv", index=False)
    print("✅ wrote", out_dir / "panel_metrics.csv")

    # ---- Panel ROC ----
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    plot_roc(ax[0, 0], yp_te, p_logreg_pca, "PCA – LogReg")
    plot_roc(ax[0, 1], yp_te, p_svm_pca, "PCA – SVM")
    ax[0, 2].set_title("PCA – VQC (pending fix)")
    ax[0, 2].axis("off")

    plot_roc(ax[1, 0], yr_te, p_logreg_r8, "Region8 – LogReg")
    plot_roc(ax[1, 1], yr_te, p_svm_r8, "Region8 – SVM")
    plot_roc(ax[1, 2], y_vqc, p_vqc, "Region8 – VQC")

    plt.tight_layout()
    out_png = out_dir / "panel_roc.png"
    plt.savefig(out_png, dpi=200)
    print("✅ wrote", out_png)


if __name__ == "__main__":
    main()