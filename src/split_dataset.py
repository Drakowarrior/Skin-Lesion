"""Minimal Stratified Dataset Splitting Utility for Skin Lesion Classification.

Performs reproducible 70% Train / 15% Validation / 15% Test stratified splitting
from data/processed/metadata.csv and generates train.csv, val.csv, test.csv.

Runnable via:
    python -m src.split_dataset [--force]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stratified Dataset Splitting Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of train.csv, val.csv, and test.csv split files",
    )
    return parser.parse_args()


def split_dataset(
    metadata_csv: Path = Path("data/processed/metadata.csv"),
    output_dir: Path = Path("data/processed"),
    force: bool = False,
) -> None:
    """Perform stratified dataset split and save train.csv, val.csv, test.csv."""
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    # Check if split files already exist
    if train_path.exists() and val_path.exists() and test_path.exists() and not force:
        print("\nSplit files already exist.\n")
        print("Use --force to regenerate the splits.")
        print("Example: python -m src.split_dataset --force\n")
        sys.exit(0)

    if not metadata_csv.exists():
        print(f"\n[ERROR] Metadata file '{metadata_csv}' not found.")
        print("Please complete Phase 2 first using: python -m src.prepare_dataset\n")
        sys.exit(1)

    df = pd.read_csv(metadata_csv)

    if len(df) == 0:
        print("\n[ERROR] Metadata CSV is empty.\n")
        sys.exit(1)

    # 1st split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    # 2nd split: 15% Val, 15% Test (50% of the 30% temp split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    # Verify Total Counts
    total_count = len(df)
    train_count = len(train_df)
    val_count = len(val_df)
    test_count = len(test_df)
    assert train_count + val_count + test_count == total_count, "Total split count mismatch!"

    # Verify Zero Overlap
    train_set = set(train_df["image_path"])
    val_set = set(val_df["image_path"])
    test_set = set(test_df["image_path"])

    overlap_tr_val = len(train_set & val_set)
    overlap_tr_test = len(train_set & test_set)
    overlap_val_test = len(val_set & test_set)
    passed_overlap = (overlap_tr_val == 0) and (overlap_tr_test == 0) and (overlap_val_test == 0)

    assert passed_overlap, "Data leakage detected: Overlap exists between dataset splits!"

    # Save Split CSV files
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["image_id", "image_path", "label"] if c in df.columns]

    train_df[cols].to_csv(train_path, index=False)
    val_df[cols].to_csv(val_path, index=False)
    test_df[cols].to_csv(test_path, index=False)

    # Calculate Class Distributions
    train_counts = train_df["label"].value_counts().to_dict()
    val_counts = val_df["label"].value_counts().to_dict()
    test_counts = test_df["label"].value_counts().to_dict()

    print("========================================")
    print("DATASET SPLIT SUMMARY")
    print("========================================")
    print(f"\nTotal Images: {total_count:,}")

    print(f"\nTRAIN\nImages: {train_count:,}\n")
    for code in TARGET_CLASSES:
        print(f"  {code:<6}: {train_counts.get(code, 0):,}")

    print(f"\nVALIDATION\nImages: {val_count:,}\n")
    for code in TARGET_CLASSES:
        print(f"  {code:<6}: {val_counts.get(code, 0):,}")

    print(f"\nTEST\nImages: {test_count:,}\n")
    for code in TARGET_CLASSES:
        print(f"  {code:<6}: {test_counts.get(code, 0):,}")

    print(f"\nOverlap Check:\n  {'PASSED' if passed_overlap else 'FAILED'}")
    print("========================================")


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()
    split_dataset(force=args.force)


if __name__ == "__main__":
    main()
