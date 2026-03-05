from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_parquet", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--vector_col", required=True, help="Name of vector feature column, e.g., z_hw or region8_z")
    ap.add_argument("--id_col", default="patch_id")
    ap.add_argument("--label_col", default="y")
    ap.add_argument("--tag_model", default="")
    ap.add_argument("--tag_feature", default="")
    return ap.parse_args()

def as_2d_array(series: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series where each element is an array/list of length D
    into a 2D numpy array of shape (N, D).
    """
    # Many parquets store vectors as lists, tuples, numpy arrays, or even strings.
    first = series.iloc[0]

    # If stored as a string like "[0.1, 0.2, ...]" (rare but happens)
    if isinstance(first, str):
        series = series.apply(lambda s: np.array(eval(s), dtype=np.float32))

    arr = np.stack(series.apply(lambda x: np.array(x, dtype=np.float32)).to_list(), axis=0)
    return arr

def main():
    args = parse_args()
    feat_path = Path(args.features_parquet)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feat_path)

    if args.vector_col not in df.columns:
        raise ValueError(f"vector_col '{args.vector_col}' not found. Available: {list(df.columns)}")

    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not found. Available: {list(df.columns)}")

    X = as_2d_array(df[args.vector_col])

    model = joblib.load(args.model_path)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model has no predict_proba(). For SVM you must train with probability=True.")

    prob = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "patch_id": df[args.id_col].astype(str).values,
        "y_true": df[args.label_col].values if args.label_col in df.columns else None,
        "prob": prob,
        "model": args.tag_model or Path(args.model_path).stem,
        "feature": args.tag_feature or args.vector_col,
    })
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} | n={len(out)} | X.shape={X.shape}")

if __name__ == "__main__":
    main()