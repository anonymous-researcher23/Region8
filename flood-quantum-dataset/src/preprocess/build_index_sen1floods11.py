from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


# Match things like:
#   USA_1082482
#   Bolivia_12345
# optionally with suffix and .tif:
#   USA_1082482_S1Hand
#   USA_1082482_S2Hand.tif
#   USA_1082482_LabelHand
SCENE_TOKEN_RE = re.compile(
    r"\b[A-Za-z]+_[0-9]+(?:_(?:S1Hand|S2Hand|LabelHand))?(?:\.tif)?\b"
)


def normalize_scene_id(token: str) -> str:
    """
    Convert:
      USA_1082482_S1Hand.tif -> USA_1082482
      USA_1082482_S2Hand     -> USA_1082482
      USA_1082482            -> USA_1082482
    """
    s = str(token).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return ""

    if s.lower().endswith(".tif"):
        s = s[:-4]

    for suf in ("_S1Hand", "_S2Hand", "_LabelHand"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


def extract_scene_id_from_row(row: pd.Series) -> str | None:
    """
    Robustly pull a scene id from any field in the split CSV row.
    We look for tokens matching SCENE_TOKEN_RE anywhere in stringified fields.
    """
    for v in row.values:
        try:
            s = str(v)
        except Exception:
            continue
        if not s or s.lower() in {"nan", "none", "null"}:
            continue
        m = SCENE_TOKEN_RE.search(s)
        if m:
            scene_id = normalize_scene_id(m.group(0))
            if scene_id:
                return scene_id
    return None


def scene_id_from_filename(tif_path: Path) -> str:
    """
    Parse base scene id from official Sen1Floods11 filenames:
      USA_1082482_S1Hand.tif  -> USA_1082482
      USA_1082482_S2Hand.tif  -> USA_1082482
      USA_1082482_LabelHand.tif -> USA_1082482
    """
    stem = tif_path.stem  # without .tif
    # remove the last suffix token (S1Hand / S2Hand / LabelHand)
    # e.g. "USA_1082482_S1Hand" -> "USA_1082482"
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def build_folder_map(folder: Path) -> dict[str, Path]:
    """
    Build mapping: base_scene_id -> tif path
    """
    m: dict[str, Path] = {}
    for tif in folder.glob("*.tif"):
        sid = scene_id_from_filename(tif)
        m[sid] = tif
    return m


def read_split_csv(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        raise ValueError(f"Empty split CSV: {csv_path}")
    df["split"] = split_name
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/processed/splits/sen1floods11_handlabeled_index.csv",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Sen1Floods11 v1.1 layout
    splits_dir = dataset_root / "splits" / "flood_handlabeled"
    hand_dir = dataset_root / "data" / "flood_events" / "HandLabeled"
    s1_dir = hand_dir / "S1Hand"
    s2_dir = hand_dir / "S2Hand"
    lb_dir = hand_dir / "LabelHand"

    for p in (splits_dir, s1_dir, s2_dir, lb_dir):
        if not p.exists():
            raise FileNotFoundError(f"Expected path not found: {p}")

    train_csv = splits_dir / "flood_train_data.csv"
    val_csv = splits_dir / "flood_valid_data.csv"
    test_csv = splits_dir / "flood_test_data.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    df_train = read_split_csv(train_csv, "train")
    df_val = read_split_csv(val_csv, "val")
    df_test = read_split_csv(test_csv, "test")
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Build fast lookup maps from actual files
    s1_map = build_folder_map(s1_dir)
    s2_map = build_folder_map(s2_dir)
    lb_map = build_folder_map(lb_dir)

    rows = []
    skipped_missing_id = 0
    skipped_missing_files = 0

    missing_id_examples = []
    missing_file_examples = []

    for _, r in df.iterrows():
        scene_id = extract_scene_id_from_row(r)
        if scene_id is None:
            skipped_missing_id += 1
            if len(missing_id_examples) < 5:
                missing_id_examples.append(dict(r))
            continue

        s1_path = s1_map.get(scene_id)
        s2_path = s2_map.get(scene_id)
        lb_path = lb_map.get(scene_id)

        if (s1_path is None) or (s2_path is None) or (lb_path is None):
            skipped_missing_files += 1
            if len(missing_file_examples) < 8:
                missing_file_examples.append(
                    {
                        "scene_id": scene_id,
                        "has_s1": s1_path is not None,
                        "has_s2": s2_path is not None,
                        "has_label": lb_path is not None,
                    }
                )
            continue

        rows.append(
            {
                "scene_id": scene_id,
                "s1_path": str(s1_path),
                "s2_path": str(s2_path),
                "label_path": str(lb_path),
                "split": r["split"],
            }
        )

    # Ensure columns exist even if empty
    out = pd.DataFrame(
        rows,
        columns=["scene_id", "s1_path", "s2_path", "label_path", "split"],
    ).drop_duplicates(subset=["scene_id", "split"])

    out.to_csv(out_csv, index=False)
    print(f"\n✅ Wrote: {out_csv}")

    if out.shape[0] == 0:
        print("\n🚨 Index CSV is EMPTY. That means we couldn't extract scene IDs from the split CSVs.")
        print(f"Skipped rows with missing/invalid scene_id: {skipped_missing_id}")
        print(f"Skipped rows with missing files after extraction: {skipped_missing_files}")

        print("\nExamples of rows where scene_id extraction failed (up to 5):")
        for ex in missing_id_examples:
            print(ex)

        print("\nExamples where scene_id extracted but files missing (up to 8):")
        for ex in missing_file_examples:
            print(ex)

        print("\nNext action:")
        print("Open one split CSV (e.g., flood_train_data.csv) and check what the identifier column contains.")
        print("If it doesn't contain tokens like 'USA_1082482' anywhere, we need to adjust extraction to that column name.")
        raise RuntimeError("Index creation produced 0 rows. See diagnostics above.")

    # Safe prints
    print("\nCounts by split:")
    print(out["split"].value_counts(dropna=False))

    print(f"\nSkipped rows with missing/invalid scene_id: {skipped_missing_id}")
    print(f"Skipped rows with missing files: {skipped_missing_files}")

    if missing_id_examples:
        print("\nExamples of rows with missing/invalid scene_id (up to 5):")
        for ex in missing_id_examples:
            print(ex)

    if missing_file_examples:
        print("\nExamples of missing files (up to 8):")
        for ex in missing_file_examples:
            print(ex)

    print("\nExample rows:")
    print(out.head(3))


if __name__ == "__main__":
    main()
