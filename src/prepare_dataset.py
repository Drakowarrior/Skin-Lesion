"""Minimal Dataset Inspection and Metadata Preparation Utility.

Inspects raw dataset images, maps class folders to the 7 target HAM10000 categories,
validates image existence, and generates data/processed/metadata.csv.

Runnable via:
    python -m src.prepare_dataset
"""

import os
from collections import Counter
from pathlib import Path

import pandas as pd

# Mapping of raw folder names to standard 7 HAM10000 class codes
CLASS_MAP = {
    "actinic keratoses": "akiec",
    "actinic keratosis": "akiec",
    "akiec": "akiec",
    "basal cell carcinoma": "bcc",
    "bcc": "bcc",
    "benign keratosis-like lesions": "bkl",
    "benign keratosis": "bkl",
    "bkl": "bkl",
    "dermatofibroma": "df",
    "df": "df",
    "melanoma": "mel",
    "mel": "mel",
    "melanocytic nevi": "nv",
    "melanocytic nevus": "nv",
    "nv": "nv",
    "vascular lesions": "vasc",
    "vasc": "vasc",
}

# Full readable class names
FULL_CLASS_NAMES = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesions",
}

TARGET_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def prepare_dataset(
    raw_dir: Path = Path("data/raw"),
    output_csv: Path = Path("data/processed/metadata.csv"),
) -> None:
    """Inspect raw dataset, validate image records, and generate processed metadata CSV."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    records = []
    missing_count = 0
    image_ids = []

    if not raw_dir.exists():
        print(f"[ERROR] Raw data directory '{raw_dir}' not found.")
        return

    # Recursively discover image files in class subfolders
    for root, _, files in os.walk(raw_dir):
        folder_name = os.path.basename(root).lower().strip()
        if folder_name in CLASS_MAP:
            class_code = CLASS_MAP[folder_name]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in image_extensions:
                    img_path = Path(root) / f
                    rel_path = img_path.relative_to(Path("."))
                    image_id = os.path.splitext(f)[0]

                    if not img_path.exists():
                        missing_count += 1
                        continue

                    records.append(
                        {
                            "image_id": image_id,
                            "image_path": str(rel_path),
                            "label": class_code,
                        }
                    )
                    image_ids.append(image_id)

    # Check for duplicate image IDs
    id_counts = Counter(image_ids)
    duplicate_count = sum(1 for count in id_counts.values() if count > 1)

    df = pd.DataFrame(records)

    # Save to data/processed/metadata.csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df[["image_id", "image_path", "label"]].to_csv(output_csv, index=False)

    class_counts = df["label"].value_counts().to_dict()

    print("========================================")
    print("DATASET PREPARATION")
    print("========================================")
    print(f"\nImages Found:   {len(df):,}")
    print(f"Valid Records:  {len(df):,}")
    print(f"Missing Images: {missing_count}")
    print(f"Duplicate IDs:  {duplicate_count}")
    print("\nClasses:\n")
    for code in TARGET_CLASSES:
        cnt = class_counts.get(code, 0)
        print(f"  {code:<6}: {cnt:>6,}")
    print("\nProcessed metadata saved to:")
    print(f"  {output_csv}")
    print("========================================")


if __name__ == "__main__":
    prepare_dataset()
