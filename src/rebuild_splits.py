"""
Rebuild Leakage-Free Dataset Splits (Phase 11)

This script:
1. Scans raw dataset images in data/raw/.
2. Computes SHA-256 hashes to group exact duplicate image content.
3. Detects and reports duplicate groups with conflicting labels.
4. Performs class-aware group-level stratified splitting (70% Train, 15% Val, 15% Test)
   using random_state = 42, keeping all files of a duplicate group strictly within one split.
5. Backs up existing split CSV files to train_old.csv, val_old.csv, test_old.csv.
6. Writes new leakage-free train.csv, val.csv, and test.csv.
7. Independently verifies zero cross-split duplicate content leakage.
"""

import argparse
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Leakage-Free Dataset Splits"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of train.csv, val.csv, and test.csv split files",
    )
    return parser.parse_args()


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def rebuild_splits(force: bool = False):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = PROCESSED_DIR / "train.csv"
    val_csv = PROCESSED_DIR / "val.csv"
    test_csv = PROCESSED_DIR / "test.csv"

    # Check if split files already exist and require --force
    if train_csv.exists() and val_csv.exists() and test_csv.exists() and not force:
        print("\nClean split files already exist.")
        print("Use --force to rebuild them.")
        print("Example: python -m src.rebuild_splits --force\n")
        sys.exit(0)

    print("========================================")
    print("REBUILDING LEAKAGE-FREE DATASET SPLITS")
    print("========================================")

    # 1. Scan images & compute SHA-256 hashes
    hash_to_files = defaultdict(list)
    hash_to_labels = defaultdict(set)

    all_file_count = 0
    for root, _, files in os.walk(RAW_DATA_DIR):
        for f in sorted(files):
            ext = Path(f).suffix.lower()
            if f.startswith(".") or ext not in SUPPORTED_EXTENSIONS:
                continue
            fp = Path(root) / f
            rel_parts = fp.relative_to(RAW_DATA_DIR).parts
            if len(rel_parts) >= 2 and rel_parts[0] in ["train", "val", "test"]:
                cls_name = rel_parts[1]
            else:
                cls_name = rel_parts[0]

            h = compute_file_sha256(fp)
            all_file_count += 1
            file_record = {
                "image_id": fp.stem,
                "image_path": str(fp),
                "label": cls_name,
                "sha256": h,
            }
            hash_to_files[h].append(file_record)
            hash_to_labels[h].add(cls_name)

    total_images = all_file_count
    unique_content_groups = len(hash_to_files)
    duplicate_groups = sum(1 for files in hash_to_files.values() if len(files) > 1)
    redundant_files = total_images - unique_content_groups

    print("\n========================================")
    print("DUPLICATE ANALYSIS")
    print("========================================")
    print(f"Total Images: {total_images}")
    print(f"Unique Content Groups: {unique_content_groups}")
    print(f"Duplicate Groups: {duplicate_groups}")
    print(f"Redundant Files: {redundant_files}")

    # 2. Check for conflicting labels in duplicate groups
    conflicting_groups = {
        h: (lbls, files)
        for h, lbls in hash_to_labels.items()
        if len(lbls) > 1
        for files in [hash_to_files[h]]
    }

    print("\n----------------------------------------")
    print("LABEL CONSISTENCY CHECK")
    print("----------------------------------------")
    if conflicting_groups:
        print(f"Notice: {len(conflicting_groups)} duplicate group(s) have conflicting folder labels:")
        for h, (lbls, files) in conflicting_groups.items():
            print(f"  - Hash {h[:8]}... Conflicting classes: {sorted(list(lbls))}")
            for f in files:
                print(f"      Path: {f['image_path']} (Label: {f['label']})")
    else:
        print("All duplicate content groups have 100% consistent class labels.")

    # 3. Build Class-Aware Duplicate Group Mapping
    class_groups = defaultdict(list)
    for h, files in hash_to_files.items():
        # Pick primary label for stratified group assignment (first file's label)
        primary_label = files[0]["label"]
        class_groups[primary_label].append(h)

    # 4. Perform Stratified Group Splitting (70% Train, 15% Val, 15% Test)
    train_records, val_records, test_records = [], [], []
    train_hashes, val_hashes, test_hashes = set(), set(), set()

    for cls in sorted(class_groups.keys()):
        groups = sorted(class_groups[cls])
        # Deterministic shuffle using random_state = 42
        random.Random(RANDOM_SEED).shuffle(groups)

        n = len(groups)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        tr_g = groups[:n_train]
        va_g = groups[n_train : n_train + n_val]
        te_g = groups[n_train + n_val :]

        for h in tr_g:
            train_hashes.add(h)
            train_records.extend(hash_to_files[h])
        for h in va_g:
            val_hashes.add(h)
            val_records.extend(hash_to_files[h])
        for h in te_g:
            test_hashes.add(h)
            test_records.extend(hash_to_files[h])

    # 5. Backup existing split CSV files if present
    print("\n----------------------------------------")
    print("OVERWRITING PROTECTION & BACKUPS")
    print("----------------------------------------")
    for name in ["train", "val", "test"]:
        orig_file = PROCESSED_DIR / f"{name}.csv"
        backup_file = PROCESSED_DIR / f"{name}_old.csv"
        if orig_file.exists() and not backup_file.exists():
            orig_file.rename(backup_file)
            print(f"Created safety backup: {backup_file}")

    # 6. Save new split CSV files
    train_df = pd.DataFrame(train_records)[["image_id", "image_path", "label"]]
    val_df = pd.DataFrame(val_records)[["image_id", "image_path", "label"]]
    test_df = pd.DataFrame(test_records)[["image_id", "image_path", "label"]]

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Saved: {train_csv} ({len(train_df)} rows)")
    print(f"Saved: {val_csv} ({len(val_df)} rows)")
    print(f"Saved: {test_csv} ({len(test_df)} rows)")

    # 7. Print Class Distribution
    print("\n----------------------------------------")
    print("CLASS DISTRIBUTION REPORT")
    print("----------------------------------------")
    all_classes = sorted(list(set(train_df["label"]).union(val_df["label"]).union(test_df["label"])))

    for split_name, df_split in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
        print(f"\n{split_name} (Total: {len(df_split)})")
        counts = Counter(df_split["label"])
        for cls in all_classes:
            print(f"  {cls:<32}: {counts[cls]}")

    # 8. Final Independent Duplicate Verification
    print("\n========================================")
    print("LEAKAGE-FREE SPLIT VERIFICATION")
    print("========================================")

    # Re-calculate SHA-256 directly from saved CSVs to verify
    def get_csv_hashes(csv_path: Path):
        df = pd.read_csv(csv_path)
        hashes = set()
        for p in df["image_path"]:
            hashes.add(compute_file_sha256(Path(p)))
        return len(df), hashes

    train_count, tr_h = get_csv_hashes(train_csv)
    val_count, va_h = get_csv_hashes(val_csv)
    test_count, te_h = get_csv_hashes(test_csv)

    tr_va_dup = len(tr_h.intersection(va_h))
    tr_te_dup = len(tr_h.intersection(te_h))
    va_te_dup = len(va_h.intersection(te_h))

    print(f"Train Images:      {train_count}")
    print(f"Validation Images: {val_count}")
    print(f"Test Images:       {test_count}")
    print(f"Total Images:      {train_count + val_count + test_count}")
    print("")
    print(f"Train ↔ Validation: {tr_va_dup}")
    print(f"Train ↔ Test:       {tr_te_dup}")
    print(f"Validation ↔ Test:  {va_te_dup}")

    if tr_va_dup == 0 and tr_te_dup == 0 and va_te_dup == 0:
        print("\n========================================")
        print("STATUS: PASS — DATASET READY FOR RESNET50 TRAINING")
        print("========================================")
        return 0
    else:
        print("\n========================================")
        print("STATUS: FAIL — DO NOT START TRAINING")
        print("========================================")
        return 1


if __name__ == "__main__":
    args = parse_args()
    sys.exit(rebuild_splits(force=args.force))
