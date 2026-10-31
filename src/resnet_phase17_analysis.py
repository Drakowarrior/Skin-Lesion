"""
ResNet50 Phase 17 Dataset & Targeted Rebalancing Analysis Script

Read-only dataset analysis script for Phase 17.
Analyzes ONLY data/processed/train.csv and data/processed/val.csv.
Computes:
1. Training and validation class counts and proportions.
2. Imbalance ratios across the 14 classes.
3. Comparison between Phase 16 weights vs Phase 17 proposed smoothed power weights.
4. Targeted minority-class oversampling projections.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.resnet_data import CLASS_LIST, CLASS_TO_IDX


TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV = Path("data/processed/val.csv")


def compute_phase17_class_weights(
    train_counts: Dict[str, int],
    smoothing: float = 20.0,
    power: float = 0.75,
    min_weight: float = 0.25,
    max_weight: float = 7.0,
) -> Dict[int, float]:
    """
    Compute Phase 17 targeted class weights using train.csv exclusively.
    
    Formula:
        effective_count = count + smoothing (20)
        raw_weight = (median_class_count / effective_count) ^ 0.75
        weight = clip(raw_weight, min_weight, max_weight)
    """
    counts_arr = np.array([train_counts[cls] for cls in CLASS_LIST], dtype=np.float32)
    median_count = np.median(counts_arr)
    
    effective_counts = counts_arr + smoothing
    raw_weights = np.power(median_count / effective_counts, power)
    clipped_weights = np.clip(raw_weights, min_weight, max_weight)
    
    return {idx: float(w) for idx, w in enumerate(clipped_weights)}


def compute_phase16_class_weights(train_counts: Dict[str, int]) -> Dict[int, float]:
    """Compute Phase 16 baseline class weights for comparison."""
    from sklearn.utils.class_weight import compute_class_weight
    df = pd.read_csv(TRAIN_CSV)
    labels = df["label"].map(CLASS_TO_IDX).values
    unique_classes = np.arange(len(CLASS_LIST))
    raw_weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    scaled_weights = np.power(raw_weights, 0.6)
    clipped_weights = np.clip(scaled_weights, 0.25, 5.0)
    return {int(cls): float(w) for cls, w in zip(unique_classes, clipped_weights)}


def run_dataset_analysis():
    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        print("ERROR: Training or validation CSV split not found.")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_total = len(train_df)
    val_total = len(val_df)

    train_counts = train_df["label"].value_counts().to_dict()
    val_counts = val_df["label"].value_counts().to_dict()

    # Fill missing classes with 0 if any
    for cls in CLASS_LIST:
        train_counts.setdefault(cls, 0)
        val_counts.setdefault(cls, 0)

    phase16_weights = compute_phase16_class_weights(train_counts)
    phase17_weights = compute_phase17_class_weights(train_counts)

    min_class_name = min(train_counts, key=train_counts.get)
    max_class_name = max(train_counts, key=train_counts.get)
    imbalance_ratio = train_counts[max_class_name] / train_counts[min_class_name] if train_counts[min_class_name] > 0 else float('inf')

    print("========================================")
    print("PHASE 17 DATASET & REBALANCING ANALYSIS")
    print("========================================")
    print(f"Total Train Images: {train_total}")
    print(f"Total Val Images:   {val_total}")
    print(f"Number of Classes:  {len(CLASS_LIST)}")
    print(f"Minimum Class:      {min_class_name} ({train_counts[min_class_name]} train samples)")
    print(f"Maximum Class:      {max_class_name} ({train_counts[max_class_name]} train samples)")
    print(f"Imbalance Ratio:    {imbalance_ratio:.2f}:1")
    print("")

    print("----------------------------------------")
    print("CLASS DISTRIBUTION & WEIGHT COMPARISON")
    print("----------------------------------------")
    print(f"{'Class Name':<30} | {'Train':<6} | {'Val':<5} | {'Phase 16 Weight':<16} | {'Phase 17 Weight':<16}")
    print("-" * 84)

    for idx, cls in enumerate(CLASS_LIST):
        tr_c = train_counts[cls]
        va_c = val_counts[cls]
        p16_w = phase16_weights[idx]
        p17_w = phase17_weights[idx]
        print(f"{cls:<30} | {tr_c:<6} | {va_c:<5} | {p16_w:<16.4f} | {p17_w:<16.4f}")

    print("-" * 84)

    print("\n----------------------------------------")
    print("TARGETED MINORITY-CLASS SUMMARY")
    print("----------------------------------------")
    target_minority = ["Dermatofibroma", "Vascular lesions", "Squamous cell carcinoma", "Actinic keratoses"]
    for cls in target_minority:
        idx = CLASS_TO_IDX[cls]
        print(f"  - {cls:<25}: Train={train_counts[cls]:<4} | P16 Weight={phase16_weights[idx]:.4f} -> P17 Weight={phase17_weights[idx]:.4f}")

    print("\n========================================")
    print("STATUS: PASS (Read-only analysis complete)")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(run_dataset_analysis())
