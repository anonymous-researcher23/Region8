#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--test_parquet", required=True)
    ap.add_argument("--feature_col", required=True)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def as_matrix(col):
    return np.stack([np.asarray(v, dtype=float) for v in col.to_list()])

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_parquet(args.train_parquet)[["patch_id","y",args.feature_col]]
    te = pd.read_parquet(args.test_parquet)[["patch_id","y",args.feature_col]]

    Xtr = as_matrix(tr[args.feature_col]); ytr = tr["y"].to_numpy().astype(int)
    Xte = as_matrix(te[args.feature_col]); yte = te["y"].to_numpy().astype(int)

    logreg = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
    svm = Pipeline([("sc", StandardScaler()), ("svm", SVC(C=5.0, gamma="scale", probability=True))])

    logreg.fit(Xtr, ytr)
    svm.fit(Xtr, ytr)

    pd.DataFrame({"patch_id": te["patch_id"], "p": logreg.predict_proba(Xte)[:,1]}).to_csv(out_dir/"logreg.csv", index=False)
    pd.DataFrame({"patch_id": te["patch_id"], "p": svm.predict_proba(Xte)[:,1]}).to_csv(out_dir/"svm.csv", index=False)

    print("✅ wrote", out_dir/"logreg.csv")
    print("✅ wrote", out_dir/"svm.csv")

if __name__ == "__main__":
    main()