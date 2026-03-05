from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# Robust CSV loading utilities
# -----------------------------
def _read_csv_robust(csv_path: str) -> pd.DataFrame:
    """
    Read prediction CSVs that might be:
      1) headered: patch_id,p_vqc
      2) headered but different columns
      3) headerless 2-col: patch_id,prob
    """
    # First try: normal header
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Fallback: headerless 2-col
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path}: expected at least 2 columns, got {df.shape[1]}")
    df = df.iloc[:, :2].copy()
    df.columns = ["patch_id", "p_vqc"]
    return df


def load_preds(
    csv_path: str,
    model: str,
    feature: str,
    *,
    flip_prob: bool = False,
    prob_override: str | None = None,
) -> pd.DataFrame:
    df = _read_csv_robust(csv_path)

    # patch id
    if "patch_id" not in df.columns:
        # some files may have the patch_id as the first column with a weird name
        df = df.rename(columns={df.columns[0]: "patch_id"})
    if "patch_id" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'patch_id'. Columns: {df.columns.tolist()}")

    # probability column
    if prob_override is not None:
        if prob_override not in df.columns:
            raise ValueError(
                f"{csv_path}: prob_override='{prob_override}' not found. Columns: {df.columns.tolist()}"
            )
        prob_col = prob_override
    else:
        prob_col = None
        # include a bunch of common names; your repo uses p_vqc a lot
        for c in ["p_vqc", "prob", "p", "p_flood", "p1", "pred_prob", "prob_flood"]:
            if c in df.columns:
                prob_col = c
                break

        # if still none and it's a 2-col headerless file we named p_vqc above, it should exist
        if prob_col is None and df.shape[1] >= 2:
            # try second column as probability as a last resort
            prob_col = df.columns[1]

        if prob_col is None:
            raise ValueError(
                f"{csv_path}: cannot find probability column. "
                f"Expected one of p_vqc/prob/p/p_flood/p1/pred_prob/prob_flood. "
                f"Columns: {df.columns.tolist()}"
            )

    # y_true (optional)
    y_col = "y_true" if "y_true" in df.columns else ("y" if "y" in df.columns else None)

    # coerce to numeric safely (handles accidental header rows mixed into data)
    prob = pd.to_numeric(df[prob_col], errors="coerce")

    # drop rows where patch_id is missing or prob is NaN
    out = pd.DataFrame(
        {
            "patch_id": df["patch_id"].astype(str),
            "prob": prob.astype(float, errors="ignore"),
            "model": model,
            "feature": feature,
        }
    )
    out = out.dropna(subset=["patch_id", "prob"]).copy()
    out["prob"] = out["prob"].astype(float)

    if flip_prob:
        out["prob"] = 1.0 - out["prob"]

    if y_col is not None:
        out["y_true"] = pd.to_numeric(df[y_col], errors="coerce").dropna().astype(int)

    return out


def get_y_true(patch_id: str) -> int | None:
    """
    Tries a couple known parquet locations. If you have different paths, add them here.
    """
    candidates = [
        "data/processed/features/sen1floods11_handlabeled/test_features.parquet",
        "data/processed/features/region8/sen1floods11_handlabeled/test.parquet",
        # add your quantum parquets too if useful:
        "data/processed/quantum_region8/sen1floods11_handlabeled/test_quantum.parquet",
    ]

    for path in candidates:
        p = Path(path)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["patch_id", "y"])
        except Exception:
            continue
        m = df[df["patch_id"].astype(str) == patch_id]
        if len(m) > 0:
            return int(m.iloc[0]["y"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_id", required=True, help="e.g., Spain_5650136_416_384")
    ap.add_argument("--image_path", required=True, help="Path to patch image (png)")
    ap.add_argument("--out_png", default="panel.png")

    # Prediction CSVs
    ap.add_argument("--logreg_pca8", default="outputs/preds/logreg_pca8_test.csv")
    ap.add_argument("--svm_pca8", default="outputs/preds/svm_pca8_test.csv")
    ap.add_argument("--vqc_pca8", default="outputs/preds/vqc_pca8_test.csv")

    ap.add_argument("--logreg_region8", default="outputs/preds/logreg_region8_test.csv")
    ap.add_argument("--svm_region8", default="outputs/preds/svm_region8_test.csv")
    ap.add_argument("--vqc_region8", default="outputs/preds/vqc_region8_test.csv")

    # Flips to fix orientation mismatches
    ap.add_argument("--flip_vqc_pca8", action="store_true", help="Use 1 - p for VQC PCA8.")
    ap.add_argument("--flip_vqc_region8", action="store_true", help="Use 1 - p for VQC Region8.")

    # Optional: show decision using thresholds
    ap.add_argument("--show_decision", action="store_true", help="Show predicted label using thresholds.")
    ap.add_argument("--thr_pca8", type=float, default=0.5, help="Threshold for PCA8 decision text.")
    ap.add_argument("--thr_region8", type=float, default=0.5, help="Threshold for Region8 decision text.")

    args = ap.parse_args()
    patch_id = args.patch_id

    # Load all prediction files
    dfs: list[pd.DataFrame] = [
        load_preds(args.logreg_pca8, "LogReg", "PCA8"),
        load_preds(args.svm_pca8, "SVM-RBF", "PCA8"),
        load_preds(args.vqc_pca8, "VQC", "PCA8", flip_prob=args.flip_vqc_pca8),
        load_preds(args.logreg_region8, "LogReg", "Region8"),
        load_preds(args.svm_region8, "SVM-RBF", "Region8"),
        load_preds(args.vqc_region8, "VQC", "Region8", flip_prob=args.flip_vqc_region8),
    ]
    all_preds = pd.concat(dfs, ignore_index=True)

    # Select this patch
    sub = all_preds[all_preds["patch_id"] == patch_id].copy()
    if sub.empty:
        raise RuntimeError(
            f"Patch id not found in prediction CSVs: {patch_id}\n"
            f"Tip: verify patch_id spelling OR check patch_id format inside your CSV.\n"
            f"Example IDs:\n{all_preds['patch_id'].head(10).tolist()}"
        )

    # Pivot to 3x2 table
    tab = sub.pivot_table(index="model", columns="feature", values="prob", aggfunc="mean")

    # Ground truth
    y_true = None
    if "y_true" in sub.columns and sub["y_true"].notna().any():
        y_true = int(sub["y_true"].dropna().iloc[0])
    else:
        y_true = get_y_true(patch_id)

    gt_str = "FLOOD" if y_true == 1 else ("NON-FLOOD" if y_true == 0 else "UNKNOWN")

    # ---- plot ----
    img = Image.open(args.image_path)

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.3])

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img)
    ax0.axis("off")
    ax0.set_title(f"Patch: {patch_id}", fontsize=12)

    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")

    # Header
    ax1.text(0.02, 0.88, "Patch (RGB + VV + VH)", fontsize=13, weight="bold")
    ax1.text(0.40, 0.72, "PCA8", fontsize=12, weight="bold")
    ax1.text(0.70, 0.72, "Region8", fontsize=12, weight="bold")

    def fmt_cell(p: float, thr: float) -> str:
        if pd.isna(p):
            return "p=NA"
        if not args.show_decision:
            return f"p={p:.3f}"
        pred = 1 if p >= thr else 0
        pred_str = "Flood" if pred == 1 else "Non-flood"
        return f"p={p:.3f} \u2192 {pred_str} (thr={thr:.3f})"

    # Rows
    row_y = {"LogReg": 0.52, "SVM-RBF": 0.34, "VQC": 0.16}
    for model in ["LogReg", "SVM-RBF", "VQC"]:
        y = row_y[model]
        ax1.text(0.02, y, f"{model}", fontsize=12, weight="bold" if model == "VQC" else None)

        p_pca = tab.loc[model, "PCA8"] if ("PCA8" in tab.columns and model in tab.index) else float("nan")
        p_reg = tab.loc[model, "Region8"] if ("Region8" in tab.columns and model in tab.index) else float("nan")

        ax1.text(0.40, y, fmt_cell(p_pca, args.thr_pca8), fontsize=11)
        ax1.text(0.70, y, fmt_cell(p_reg, args.thr_region8), fontsize=11)

    ax1.text(0.02, -0.02, f"Ground truth: {gt_str}", fontsize=12, weight="bold")

    # Small note if flips are enabled
    flip_note = []
    if args.flip_vqc_pca8:
        flip_note.append("VQC-PCA8 flipped")
    if args.flip_vqc_region8:
        flip_note.append("VQC-Region8 flipped")
    if flip_note:
        ax1.text(0.70, -0.02, " | ".join(flip_note), fontsize=9)

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved panel: {out_path}")


if __name__ == "__main__":
    main()