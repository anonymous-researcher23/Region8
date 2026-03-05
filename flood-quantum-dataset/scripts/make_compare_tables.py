import argparse, json
import pandas as pd
import numpy as np

def load_hw(run_dir):
    df = pd.read_csv(f"{run_dir}/predictions.csv")
    with open(f"{run_dir}/metrics.json","r") as f:
        m = json.load(f)
    thr = float(m.get("thr", 0.5))
    df["pred_hw"] = (df["p1"] >= thr).astype(int)
    df.rename(columns={"p1":"p_hw"}, inplace=True)
    return df, thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw_run", required=True)
    ap.add_argument("--quantum_test", required=True, help="test_quantum.parquet")
    ap.add_argument("--out_csv", default="outputs/compare_hw_examples.csv")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    hw, thr = load_hw(args.hw_run)
    qt = pd.read_parquet(args.quantum_test)

    # join on scene_id/patch_id/row/col (more robust than index)
    keys = ["scene_id","patch_id","row","col"]
    df = hw.merge(qt[keys + ["y","flood_frac","z_hw","theta_hw"]], on=keys, how="left")
    df["y_true"] = df["y_true"].astype(int)

    # categories
    def cat(r):
        y=r["y_true"]; p=r["pred_hw"]
        if y==1 and p==1: return "TP"
        if y==0 and p==1: return "FP"
        if y==1 and p==0: return "FN"
        return "TN"

    df["cat"] = df.apply(cat, axis=1)

    # rank examples: FP with highest p, FN with lowest p, etc.
    parts=[]
    for c in ["TP","FP","FN","TN"]:
        sub=df[df["cat"]==c].copy()
        if c in ["TP","FP"]:
            sub=sub.sort_values("p_hw", ascending=False)
        else:
            sub=sub.sort_values("p_hw", ascending=True)
        parts.append(sub.head(args.k))

    out=pd.concat(parts, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv, "| thr:", thr)

if __name__=="__main__":
    main()
