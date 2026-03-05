#!/usr/bin/env bash
set -euo pipefail

# ---- Inputs ----
BASE_DIR="outputs/all_models_3x3"
OUT_DIR="outputs/figs"
VQC_PCA8="outputs/preds/vqc_pca8_test.csv"
VQC_REGION8="outputs/preds/vqc_region8_test.csv"

# Thresholds (edit if you regenerate proper matching VAL for PCA8)
THR_PCA8="0.8600359"      # visual threshold (median from test distribution)
THR_REGION8="0.4564493"   # from val_fromrun

# ---- Checks ----
for f in "$VQC_PCA8" "$VQC_REGION8"; do
  [[ -f "$f" ]] || { echo "Missing preds file: $f"; exit 1; }
done
[[ -d "$BASE_DIR" ]] || { echo "Missing base dir: $BASE_DIR"; exit 1; }

mkdir -p "$OUT_DIR"/{TP,FP,FN,TN}

# ---- Worker ----
make_one () {
  local img="$1"
  local split="$2"

  local base
  base="$(basename "$img" .png)"

  # Expect filenames like: 0009_Spain_5650136_288_160.png
  # patch_id = everything after first underscore
  local patch_id
  patch_id="$(echo "$base" | sed -E 's/^[0-9]+_//')"

  if [[ -z "$patch_id" || "$patch_id" == "$base" ]]; then
    echo "[SKIP] Can't parse patch_id from $img"
    return 0
  fi

  local out_png="$OUT_DIR/$split/panel_${patch_id}.png"

  PYTHONPATH=. python scripts/make_patch_panel.py \
    --patch_id "$patch_id" \
    --image_path "$img" \
    --vqc_pca8 "$VQC_PCA8" \
    --vqc_region8 "$VQC_REGION8" \
    --show_decision \
    --thr_pca8 "$THR_PCA8" \
    --thr_region8 "$THR_REGION8" \
    --out_png "$out_png" >/dev/null

  echo "[OK] $split  $patch_id"
}

export -f make_one

# ---- Loop (serial) ----
for split in TP FP FN TN; do
  dir="$BASE_DIR/$split"
  [[ -d "$dir" ]] || { echo "[WARN] Missing $dir (skipping)"; continue; }

  echo "=== Processing $split ==="
  shopt -s nullglob
  imgs=("$dir"/*.png)
  shopt -u nullglob

  if [[ ${#imgs[@]} -eq 0 ]]; then
    echo "[WARN] No PNGs in $dir"
    continue
  fi

  for img in "${imgs[@]}"; do
    make_one "$img" "$split"
  done
done

echo "Done. Panels saved under $OUT_DIR/{TP,FP,FN,TN}/"
