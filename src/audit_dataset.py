"""
Dataset Audit Script for Skin Lesions Major Project (Phase 9)

Performs a comprehensive, read-only audit of the dataset located in data/raw/.
Recursively inspects image files, folder structure, image integrity, class distribution,
duplicate files, and unexpected non-image files.
"""

import sys
import os
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image


RAW_DATA_DIR = Path("data/raw")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 14 classes identified in data/raw
EXPECTED_CLASSES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "HFMD",
    "Healthy",
    "Measles",
    "Melanocytic nevi",
    "Melanoma",
    "Monkeypox",
    "Squamous cell carcinoma",
    "Vascular lesions"
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for exact duplicate content detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset():
    if not RAW_DATA_DIR.exists():
        print(f"ERROR: Dataset directory '{RAW_DATA_DIR}' does not exist.")
        sys.exit(1)

    print("========================================")
    print("DATASET AUDIT")
    print("========================================")

    all_files = []
    unexpected_files = []
    empty_folders = []
    
    # Class tracking
    class_total_counts = Counter()
    class_split_counts = defaultdict(Counter) # split -> class -> count
    
    # Image integrity tracking
    valid_images = 0
    corrupt_images = []
    image_formats = Counter()
    image_dimensions = Counter()
    
    # Hash / duplicate tracking
    file_hashes = defaultdict(list) # sha256 -> list of path strings
    filename_map = defaultdict(list) # filename -> list of path strings
    
    # Walk data/raw
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        root_path = Path(root)
        
        # Check empty directory (excluding root itself)
        if root_path != RAW_DATA_DIR and len(dirs) == 0 and len(files) == 0:
            empty_folders.append(root_path)
            
        for file in sorted(files):
            file_path = root_path / file
            
            # Skip hidden metadata/system files from unexpected list count if standard, but track them
            if file.startswith(".") or file in ["README.md", ".gitkeep"]:
                unexpected_files.append(file_path)
                continue
                
            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                unexpected_files.append(file_path)
                continue

            # Determine relative path components
            rel_path = file_path.relative_to(RAW_DATA_DIR)
            parts = rel_path.parts
            
            # Check split vs direct class folder
            if len(parts) >= 2 and parts[0] in ["train", "val", "test"]:
                split = parts[0]
                class_name = parts[1]
            elif len(parts) >= 1:
                split = "root"
                class_name = parts[0]
            else:
                split = "unknown"
                class_name = "unknown"

            all_files.append(file_path)
            class_total_counts[class_name] += 1
            class_split_counts[split][class_name] += 1
            
            # Filename tracking
            filename_map[file_path.name].append(str(file_path))
            
            # Check Image Integrity
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify file headers/structure
                
                # Re-open to read dimensions and format (verify closes file/resets stream)
                with Image.open(file_path) as img:
                    fmt = img.format or ext[1:].upper()
                    w, h = img.size
                    image_formats[fmt] += 1
                    image_dimensions[(w, h)] += 1
                    valid_images += 1
                    
            except Exception as e:
                corrupt_images.append((file_path, str(e)))

            # Compute content hash for duplicate detection
            file_hash = compute_sha256(file_path)
            file_hashes[file_hash].append(str(file_path))

    # Calculate statistics
    total_images = len(all_files)
    total_classes = len(class_total_counts)
    
    # Hash duplicates
    duplicate_hash_groups = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
    total_duplicate_content_files = sum(len(paths) - 1 for paths in duplicate_hash_groups.values())
    
    # Duplicate filenames across splits/folders
    duplicate_name_groups = {name: paths for name, paths in filename_map.items() if len(paths) > 1}
    total_duplicate_filenames = sum(len(paths) - 1 for paths in duplicate_name_groups.values())

    # Print Summary Results
    print(f"\nTotal Images: {total_images}")
    print(f"Classes: {total_classes}")
    if total_images > 0:
        min_class = min(class_total_counts.items(), key=lambda x: x[1])
        max_class = max(class_total_counts.items(), key=lambda x: x[1])
        imbalance_ratio = max_class[1] / min_class[1] if min_class[1] > 0 else float('inf')
        print(f"Minimum Class Size: {min_class[0]} ({min_class[1]} images)")
        print(f"Maximum Class Size: {max_class[0]} ({max_class[1]} images)")
        print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1")

    print("\n----------------------------------------")
    print("CLASS DISTRIBUTION")
    print("----------------------------------------")
    print(f"{'Class Name':<32} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("-" * 72)
    
    sorted_classes = sorted(class_total_counts.keys())
    for cls in sorted_classes:
        train_c = class_split_counts["train"][cls]
        val_c = class_split_counts["val"][cls]
        test_c = class_split_counts["test"][cls]
        tot_c = class_total_counts[cls]
        print(f"{cls:<32} | {train_c:<7} | {val_c:<7} | {test_c:<7} | {tot_c:<7}")
        
    print("-" * 72)
    print(f"{'TOTAL':<32} | {sum(class_split_counts['train'].values()):<7} | {sum(class_split_counts['val'].values()):<7} | {sum(class_split_counts['test'].values()):<7} | {total_images:<7}")

    print("\n----------------------------------------")
    print("IMAGE INTEGRITY & METADATA")
    print("----------------------------------------")
    print(f"Valid / Readable Images: {valid_images} / {total_images}")
    print(f"Corrupt / Unreadable Images: {len(corrupt_images)}")
    if corrupt_images:
        print("  Corrupt Files List:")
        for path, err in corrupt_images:
            print(f"    - {path}: {err}")

    print("\nImage Format Distribution:")
    for fmt, count in image_formats.most_common():
        pct = (count / total_images * 100) if total_images > 0 else 0
        print(f"  - {fmt}: {count} ({pct:.1f}%)")

    print("\nImage Dimension Summary (Top 5):")
    for dim, count in image_dimensions.most_common(5):
        pct = (count / total_images * 100) if total_images > 0 else 0
        print(f"  - {dim[0]}x{dim[1]} px: {count} ({pct:.1f}%)")
    print(f"Total Unique Image Dimensions: {len(image_dimensions)}")

    print("\n----------------------------------------")
    print("DATA LEAKAGE & DUPLICATES REPORT")
    print("----------------------------------------")
    print(f"Duplicate Image Content (Exact Hash Matches): {len(duplicate_hash_groups)} groups ({total_duplicate_content_files} redundant files)")
    print(f"Duplicate Filenames Across Folders: {len(duplicate_name_groups)} groups ({total_duplicate_filenames} repeated filenames)")

    print("\n----------------------------------------")
    print("UNEXPECTED FILES & EMPTY FOLDERS")
    print("----------------------------------------")
    print(f"Empty Folders Found: {len(empty_folders)}")
    print(f"Unexpected Files Found: {len(unexpected_files)}")
    if unexpected_files:
        print("  Sample unexpected files (e.g. system files / metadata):")
        for u in unexpected_files[:5]:
            print(f"    - {u}")

    print("\n----------------------------------------")
    print("FINAL CLASSIFICATION SCOPE")
    print("----------------------------------------")
    print("Scope: 14-Class Combined Dermatological & Viral Lesion Dataset")
    print("Classes:")
    for idx, cls in enumerate(sorted_classes, 1):
        print(f"  {idx:2d}. {cls}")

    print("\n========================================")
    if len(corrupt_images) == 0 and total_images > 0:
        print("STATUS: PASS")
        print("========================================")
        return 0
    else:
        print("STATUS: FAIL (Corrupt images found or zero images loaded)")
        print("========================================")
        return 1


if __name__ == "__main__":
    sys.exit(audit_dataset())
