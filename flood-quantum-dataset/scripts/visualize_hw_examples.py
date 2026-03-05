import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_patch(x_path: str) -> np.ndarray:
    """
    Load a patch tensor from disk.
    Supports:
      - .npy (array)
      - .npz (expects key 'x' or first array)
    Returns: float32 array, shape either (C,H,W) or (H,W,C)
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"x_path not found: {x_path}")

    if x_path.endswith(".npy"):
        x = np.load(x_path)
    elif x_path.endswith(".npz"):
        z = np.load(x_path)
        if "x" in z:
            x = z["x"]
        else:
            x = z[z.files[0]]
    else:
        raise ValueError(f"Unsupported patch file type: {x_path}")

    return np.asarray(x, dtype=np.float32)


def to_chw(x: np.ndarray) -> np.ndarray:
    """Ensure shape is (C,H,W)."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {x.shape}")

    # If (H,W,C) -> (C,H,W)
    # heuristic: last dim small (#channels)
    if x.shape[-1] <= 16 and x.shape[0] >= 16 and x.shape[1] >= 16:
        x = np.transpose(x, (2, 0, 1))

    return x


def norm01(a: np.ndarray) -> np.ndarray:
    """Robust normalize to [0,1] for display (percentile clip)."""
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        return np.zeros_like(a)
    a = (a - lo) / (hi - lo)
    return np.clip(a, 0, 1)


def save_example_figure(out_path: str, x_chw: np.ndarray, meta: dict):
    """
    Expected band order:
      [VV, VH, B2, B3, B4, B8]  (your project spec)
    RGB uses [B4, B3, B2] = indices [4,3,2]
    """
    if x_chw.shape[0] < 5:
        raise ValueError(f"Expected >=5 channels for VV,VH,B2,B3,B4... got {x_chw.shape}")

    vv = x_chw[0]
    vh = x_chw[1]
    b2 = x_chw[2]
    b3 = x_chw[3]
    b4 = x_chw[4]

    rgb = np.stack([b4, b3, b2], axis=-1)
    rgb = norm01(rgb)

    fig = plt.figure(figsize=(10, 3.2))
    gs = fig.add_gridspec(1, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(rgb)
    ax0.set_title("Sentinel-2 RGB (B4,B3,B2)")
    ax0.axis("off")

    ax1.imshow(norm01(vv), cmap="gray")
    ax1.set_title("SAR VV")
    ax1.axis("off")

    ax2.imshow(norm01(vh), cmap="gray")
    ax2.set_title("SAR VH")
    ax2.axis("off")

    title = (
        f"{meta['cat']} | scene={meta['scene_id']} patch={meta['patch_id']} (r{meta['row']},c{meta['col']})\n"
        f"y_true={meta['y_true']}  pred={meta['pred_hw']}  p_hw={meta['p_hw']:.3f}  thr={meta['thr']:.2f}  "
        f"flood_frac={meta['flood_frac']:.3f}"
    )
    fig.suptitle(title, fontsize=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def pick_flood_frac_column(df: pd.DataFrame) -> str | None:
    """
    Your compare_hw_examples.csv may have flood_frac_x / flood_frac_y after merges.
    Pick the best available column.
    """
    if "flood_frac" in df.columns:
        return "flood_frac"
    if "flood_frac_x" in df.columns:
        return "flood_frac_x"
    if "flood_frac_y" in df.columns:
        return "flood_frac_y"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples_csv", required=True, help="outputs/compare_hw_examples.csv")
    ap.add_argument("--patch_parquet", required=True, help=".../patches/.../test.parquet (has x_path)")
    ap.add_argument("--out_dir", default="outputs/hw_gallery")
    ap.add_argument("--max_per_cat", type=int, default=10, help="limit examples per TP/FP/FN/TN")
    args = ap.parse_args()

    ex = pd.read_csv(args.examples_csv)

    # required columns (flood_frac handled separately)
    required = ["scene_id", "patch_id", "row", "col", "cat", "y_true", "pred_hw", "p_hw"]
    for c in required:
        if c not in ex.columns:
            raise ValueError(f"examples_csv missing column: {c}. Got: {list(ex.columns)}")

    flood_col = pick_flood_frac_column(ex)
    if flood_col is None:
        # not fatal, but we’ll show NaN
        ex["flood_frac_use"] = np.nan
    else:
        ex["flood_frac_use"] = ex[flood_col]

    # patch parquet provides x_path
    patches = pd.read_parquet(args.patch_parquet)[["scene_id", "patch_id", "row", "col", "x_path"]]
    df = ex.merge(patches, on=["scene_id", "patch_id", "row", "col"], how="left")

    if df["x_path"].isna().any():
        missing = df[df["x_path"].isna()].head(10)
        raise RuntimeError(
            "Some examples missing x_path after merge. "
            "This means keys didn't match between examples and patches.\n"
            f"Sample missing rows:\n{missing}"
        )

    # threshold: if you didn’t store it in examples, default to 0.5
    thr = 0.5
    if "thr" in df.columns:
        try:
            thr = float(df["thr"].iloc[0])
        except Exception:
            pass
    df["thr"] = thr

    # limit examples per category
    outs = []
    for cat in ["TP", "FP", "FN", "TN"]:
        sub = df[df["cat"] == cat].head(args.max_per_cat)
        outs.append(sub)
    df = pd.concat(outs, ignore_index=True)

    print("Saving gallery examples:", len(df))
    for i, r in df.iterrows():
        x = load_patch(r["x_path"])
        x = to_chw(x)

        out_path = os.path.join(
            args.out_dir,
            str(r["cat"]),
            f"{i:04d}_{r['scene_id']}_{int(r['row'])}_{int(r['col'])}.png",
        )

        meta = {
            "cat": str(r["cat"]),
            "scene_id": str(r["scene_id"]),
            "patch_id": str(r["patch_id"]),
            "row": int(r["row"]),
            "col": int(r["col"]),
            "y_true": int(r["y_true"]),
            "pred_hw": int(r["pred_hw"]),
            "p_hw": float(r["p_hw"]),
            "thr": float(r["thr"]),
            "flood_frac": float(r["flood_frac_use"]) if not pd.isna(r["flood_frac_use"]) else float("nan"),
        }

        save_example_figure(out_path, x, meta)

    print("Done. Gallery saved under:", args.out_dir)
    print("Folders: TP / FP / FN / TN")


if __name__ == "__main__":
    main()
