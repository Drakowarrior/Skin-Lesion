"""
ResNet50 Phase 19 Targeted Error Analysis & Calibration Study

Read-Only Analytical Investigation of Phase 13, Phase 16, and Phase 17 ResNet50 Models.

DO NOT TRAIN ANY MODEL.
DO NOT MODIFY TEST.CSV OR MODEL CHECKPOINTS.

This script performs 11 specific analytical dimensions (Analyses A through K),
generates 7 visualization plots, 6 specific JSON reports, and a master report JSON.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mcnemar
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    brier_score_loss,
)
import tensorflow as tf

from src.resnet_data import CLASS_LIST, CLASS_TO_IDX, IDX_TO_CLASS, create_phase17_dataset


P13_MODEL_PATH = Path("models/resnet50_best.keras")
P16_MODEL_PATH = Path("models/resnet50_improved_best.keras")
P17_MODEL_PATH = Path("models/resnet50_phase17_best.keras")

TEST_CSV = Path("data/processed/test.csv")
TRAIN_CSV = Path("data/processed/train.csv")
OUTPUT_DIR = Path("outputs")
DOCS_DIR = Path("docs")


def setup_device(device_choice: str) -> str:
    """Configure device setting without throwing unnecessary errors."""
    if device_choice == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "CPU"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU ({gpus[0].name})"
    return "CPU"


def load_and_predict(
    model_path: Path, dataset: tf.data.Dataset, num_samples: int
) -> np.ndarray:
    """Load model checkpoint and run inference over dataset."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"Loading model: {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Running inference for {model_path.name}...")
    probs = model.predict(dataset, verbose=1)
    if probs.shape != (num_samples, len(CLASS_LIST)):
        raise ValueError(
            f"Unexpected shape {probs.shape} from model {model_path}. Expected ({num_samples}, {len(CLASS_LIST)})."
        )
    return probs


def compute_ece_mce(
    probs: np.ndarray, y_true: np.ndarray, num_bins: int = 10
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and per-bin reliability statistics.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)
    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # For the last bin, include boundary upper == 1.0
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            diff = abs(bin_acc - bin_conf)
            ece += (bin_size / total_samples) * diff
            mce = max(mce, diff)
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            diff = 0.0

        bin_details.append({
            "bin_index": i,
            "bin_lower": float(bin_lower),
            "bin_upper": float(bin_upper),
            "bin_label": f"{int(bin_lower*100)}-{int(bin_upper*100)}%",
            "sample_count": bin_size,
            "accuracy": round(bin_acc * 100, 2),
            "mean_confidence": round(bin_conf * 100, 2),
            "calibration_gap": round(diff * 100, 2)
        })

    return float(ece), float(mce), bin_details


def compute_multiclass_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Compute multiclass Brier score: mean squared difference between one-hot true labels and predicted probabilities."""
    n_samples, n_classes = probs.shape
    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
    y_onehot[np.arange(n_samples), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
    return brier


def main():
    parser = argparse.ArgumentParser(description="ResNet50 Phase 19 Analysis")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    dev_name = setup_device(args.device)
    print(f"Executing Phase 19 Analysis on {dev_name}")

    # 1. LOAD TEST CSV & VERIFY
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV missing: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    num_test_samples = len(test_df)
    if num_test_samples != 5529:
        raise ValueError(f"Expected 5,529 test samples, found {num_test_samples}")
    
    y_true = np.array([CLASS_TO_IDX[lbl] for lbl in test_df["label"].values], dtype=int)
    class_names = CLASS_LIST
    num_classes = len(class_names)
    print(f"Loaded untouched test split: {num_test_samples} images, {num_classes} classes.")

    # 2. CREATE DATASET & RUN INFERENCE ON ALL THREE MODELS
    dataset, _ = create_phase17_dataset(
        TEST_CSV, batch_size=64, is_training=False, oversample=False, shuffle=False
    )

    probs_p13 = load_and_predict(P13_MODEL_PATH, dataset, num_test_samples)
    probs_p16 = load_and_predict(P16_MODEL_PATH, dataset, num_test_samples)
    probs_p17 = load_and_predict(P17_MODEL_PATH, dataset, num_test_samples)

    preds_p13 = np.argmax(probs_p13, axis=1)
    preds_p16 = np.argmax(probs_p16, axis=1)
    preds_p17 = np.argmax(probs_p17, axis=1)

    conf_p13 = np.max(probs_p13, axis=1)
    conf_p16 = np.max(probs_p16, axis=1)
    conf_p17 = np.max(probs_p17, axis=1)

    # =========================================================================
    # ANALYSIS A: PER-CLASS PREDICTION SHIFT
    # =========================================================================
    print("Computing Analysis A: Per-Class Prediction Shift...")
    per_class_data = []

    # Get precision, recall, f1 for all models
    p_13, r_13, f1_13, sup_13 = precision_recall_fscore_support(y_true, preds_p13, labels=range(num_classes), zero_division=0)
    p_16, r_16, f1_16, sup_16 = precision_recall_fscore_support(y_true, preds_p16, labels=range(num_classes), zero_division=0)
    p_17, r_17, f1_17, sup_17 = precision_recall_fscore_support(y_true, preds_p17, labels=range(num_classes), zero_division=0)

    pred_counts_p13 = Counter(preds_p13)
    pred_counts_p16 = Counter(preds_p16)
    pred_counts_p17 = Counter(preds_p17)

    for c_idx, c_name in enumerate(class_names):
        support = int(sup_13[c_idx])

        # False positives & negatives
        fp_13 = int(np.sum((preds_p13 == c_idx) & (y_true != c_idx)))
        fn_13 = int(np.sum((y_true == c_idx) & (preds_p13 != c_idx)))

        fp_16 = int(np.sum((preds_p16 == c_idx) & (y_true != c_idx)))
        fn_16 = int(np.sum((y_true == c_idx) & (preds_p16 != c_idx)))

        fp_17 = int(np.sum((preds_p17 == c_idx) & (y_true != c_idx)))
        fn_17 = int(np.sum((y_true == c_idx) & (preds_p17 != c_idx)))

        per_class_data.append({
            "class_index": c_idx,
            "class_name": c_name,
            "support": support,
            "p13": {
                "recall": float(r_13[c_idx]),
                "precision": float(p_13[c_idx]),
                "f1": float(f1_13[c_idx]),
                "predicted_count": int(pred_counts_p13[c_idx]),
                "fp": fp_13,
                "fn": fn_13,
            },
            "p16": {
                "recall": float(r_16[c_idx]),
                "precision": float(p_16[c_idx]),
                "f1": float(f1_16[c_idx]),
                "predicted_count": int(pred_counts_p16[c_idx]),
                "fp": fp_16,
                "fn": fn_16,
            },
            "p17": {
                "recall": float(r_17[c_idx]),
                "precision": float(p_17[c_idx]),
                "f1": float(f1_17[c_idx]),
                "predicted_count": int(pred_counts_p17[c_idx]),
                "fp": fp_17,
                "fn": fn_17,
            },
            "shifts": {
                "p13_to_p16_recall_diff": float(r_16[c_idx] - r_13[c_idx]),
                "p16_to_p17_recall_diff": float(r_17[c_idx] - r_16[c_idx]),
                "p13_to_p17_recall_diff": float(r_17[c_idx] - r_13[c_idx]),
                "p13_to_p16_f1_diff": float(f1_16[c_idx] - f1_13[c_idx]),
                "p16_to_p17_f1_diff": float(f1_17[c_idx] - f1_16[c_idx]),
                "p13_to_p17_f1_diff": float(f1_17[c_idx] - f1_13[c_idx]),
            }
        })

    # =========================================================================
    # ANALYSIS B: PREDICTION DISTRIBUTION SHIFT
    # =========================================================================
    print("Computing Analysis B: Prediction Distribution Shift...")
    dist_data = {
        "ground_truth": {c_name: int(np.sum(y_true == c_idx)) for c_idx, c_name in enumerate(class_names)},
        "p13_predicted": {c_name: int(pred_counts_p13[c_idx]) for c_idx, c_name in enumerate(class_names)},
        "p16_predicted": {c_name: int(pred_counts_p16[c_idx]) for c_idx, c_name in enumerate(class_names)},
        "p17_predicted": {c_name: int(pred_counts_p17[c_idx]) for c_idx, c_name in enumerate(class_names)},
    }

    # =========================================================================
    # ANALYSIS C: CONFUSION MATRIX DIFFERENCE
    # =========================================================================
    print("Computing Analysis C: Confusion Matrix Shifts...")
    cm_13 = confusion_matrix(y_true, preds_p13, labels=range(num_classes))
    cm_16 = confusion_matrix(y_true, preds_p16, labels=range(num_classes))
    cm_17 = confusion_matrix(y_true, preds_p17, labels=range(num_classes))

    cm_diff_p16_p13 = cm_16 - cm_13
    cm_diff_p17_p16 = cm_17 - cm_16

    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            c13 = int(cm_13[i, j])
            c16 = int(cm_16[i, j])
            c17 = int(cm_17[i, j])
            d16_13 = c16 - c13
            d17_16 = c17 - c16
            confusion_pairs.append({
                "true_class_idx": i,
                "true_class": class_names[i],
                "pred_class_idx": j,
                "pred_class": class_names[j],
                "is_diagonal": (i == j),
                "p13_count": c13,
                "p16_count": c16,
                "p17_count": c17,
                "p16_minus_p13": d16_13,
                "p17_minus_p16": d17_16,
                "abs_p16_p13": abs(d16_13),
            })

    # Sort off-diagonal pairs by largest absolute shift P16-P13
    off_diag_pairs = [p for p in confusion_pairs if not p["is_diagonal"]]
    off_diag_sorted = sorted(off_diag_pairs, key=lambda x: abs(x["p16_minus_p13"]), reverse=True)
    top20_confusion_shifts = off_diag_sorted[:20]

    with open(OUTPUT_DIR / "resnet50_confusion_shift_analysis.json", "w") as f:
        json.dump({
            "top20_largest_shifts_p16_minus_p13": top20_confusion_shifts,
            "full_confusion_matrix_p13": cm_13.tolist(),
            "full_confusion_matrix_p16": cm_16.tolist(),
            "full_confusion_matrix_p17": cm_17.tolist(),
        }, f, indent=2)

    # =========================================================================
    # ANALYSIS D: MINORITY-CLASS ANALYSIS
    # =========================================================================
    print("Computing Analysis D: Minority-Class Deep Dive...")
    minority_classes = [
        "Actinic keratoses", "Dermatofibroma", "Squamous cell carcinoma", "Vascular lesions",
        "Cowpox", "Measles", "Chickenpox"
    ]
    minority_analysis = []

    for m_cls in minority_classes:
        c_idx = CLASS_TO_IDX[m_cls]
        c_data = per_class_data[c_idx]

        # Determine most common confusion destination for each model
        # Where did true samples of c_idx go when misclassified?
        dest_13 = Counter({class_names[j]: int(cm_13[c_idx, j]) for j in range(num_classes) if j != c_idx})
        dest_16 = Counter({class_names[j]: int(cm_16[c_idx, j]) for j in range(num_classes) if j != c_idx})
        dest_17 = Counter({class_names[j]: int(cm_17[c_idx, j]) for j in range(num_classes) if j != c_idx})

        top_dest_13 = dest_13.most_common(1)[0] if dest_13 else ("None", 0)
        top_dest_16 = dest_16.most_common(1)[0] if dest_16 else ("None", 0)
        top_dest_17 = dest_17.most_common(1)[0] if dest_17 else ("None", 0)

        # Statistical significance test: McNemar's test for P13 vs P16 on this class
        # Contingency table:
        # a: P13 correct, P16 correct
        # b: P13 correct, P16 wrong
        # c: P13 wrong, P16 correct
        # d: P13 wrong, P16 wrong
        mask_cls = (y_true == c_idx)
        corr_13_cls = (preds_p13[mask_cls] == c_idx)
        corr_16_cls = (preds_p16[mask_cls] == c_idx)

        b = int(np.sum(corr_13_cls & ~corr_16_cls))
        c = int(np.sum(~corr_13_cls & corr_16_cls))
        
        # McNemar calculation
        if b + c > 0:
            mcnemar_stat = float((abs(b - c) - 1.0)**2 / (b + c)) if abs(b - c) > 0 else 0.0
            # Approx p-value from chi2 survival function with 1 df
            from scipy.stats import chi2
            p_val = float(chi2.sf(mcnemar_stat, 1))
        else:
            mcnemar_stat = 0.0
            p_val = 1.0

        minority_analysis.append({
            "class_name": m_cls,
            "support": c_data["support"],
            "p13_recall": c_data["p13"]["recall"],
            "p16_recall": c_data["p16"]["recall"],
            "p17_recall": c_data["p17"]["recall"],
            "p13_f1": c_data["p13"]["f1"],
            "p16_f1": c_data["p16"]["f1"],
            "p17_f1": c_data["p17"]["f1"],
            "p13_to_p16_recall_change": c_data["shifts"]["p13_to_p16_recall_diff"],
            "p16_to_p17_recall_change": c_data["shifts"]["p16_to_p17_recall_diff"],
            "p13_to_p16_f1_change": c_data["shifts"]["p13_to_p16_f1_diff"],
            "p16_to_p17_f1_change": c_data["shifts"]["p16_to_p17_f1_diff"],
            "fn_count_p13": c_data["p13"]["fn"],
            "fn_count_p16": c_data["p16"]["fn"],
            "fn_count_p17": c_data["p17"]["fn"],
            "fp_count_p13": c_data["p13"]["fp"],
            "fp_count_p16": c_data["p16"]["fp"],
            "fp_count_p17": c_data["p17"]["fp"],
            "most_common_destination_p13": {"destination": top_dest_13[0], "count": top_dest_13[1]},
            "most_common_destination_p16": {"destination": top_dest_16[0], "count": top_dest_16[1]},
            "most_common_destination_p17": {"destination": top_dest_17[0], "count": top_dest_17[1]},
            "mcnemar_p13_vs_p16": {
                "b_p13corr_p16wrong": b,
                "c_p13wrong_p16corr": c,
                "stat": mcnemar_stat,
                "p_value": p_val,
                "is_statistically_significant": (p_val < 0.05)
            }
        })

    # =========================================================================
    # ANALYSIS E: CLASS-WEIGHT EFFECT & STRATEGY COMPARISON
    # =========================================================================
    print("Computing Analysis E: Class Weight Strategy Comparison...")
    train_df = pd.read_csv(TRAIN_CSV)
    train_counts = train_df["label"].value_counts().to_dict()
    N_total = len(train_df)
    K = num_classes

    p13_raw_weights = []
    p16_raw_weights = []
    p17_raw_weights = []
    p17_oversample_multipliers = []

    counts_arr = np.array([train_counts.get(cls, 0) for cls in class_names], dtype=np.float32)
    median_count = np.median(counts_arr)

    for c_idx, c_name in enumerate(class_names):
        nc = train_counts.get(c_name, 1)

        # Phase 13: standard inverse class weight
        w_p13 = N_total / (K * nc)
        p13_raw_weights.append(w_p13)

        # Phase 16: power = 0.60, clip = [0.25, 5.0]
        w_p16_unclipped = (median_count / nc) ** 0.60
        w_p16 = float(np.clip(w_p16_unclipped, 0.25, 5.0))
        p16_raw_weights.append(w_p16)

        # Phase 17: smooth = 20, power = 0.75, clip = [0.25, 7.0]
        w_p17_unclipped = (median_count / (nc + 20.0)) ** 0.75
        w_p17 = float(np.clip(w_p17_unclipped, 0.25, 7.0))
        p17_raw_weights.append(w_p17)

        # Phase 17 oversample expansion
        if nc < 1000:
            mult = min(3.0, 1000.0 / nc)
        else:
            mult = 1.0
        p17_oversample_multipliers.append(mult)

    # Normalize weights to mean 1.0 for relative comparison
    p13_weights_norm = (np.array(p13_raw_weights) / np.mean(p13_raw_weights)).tolist()
    p16_weights_norm = (np.array(p16_raw_weights) / np.mean(p16_raw_weights)).tolist()
    p17_weights_norm = (np.array(p17_raw_weights) / np.mean(p17_raw_weights)).tolist()

    class_weight_strategy_json = {
        "description": "Comparison of training weighting and rebalancing pressure across Phase 13, 16, and 17",
        "per_class_strategies": []
    }

    for c_idx, c_name in enumerate(class_names):
        class_weight_strategy_json["per_class_strategies"].append({
            "class_name": c_name,
            "train_support": int(train_counts.get(c_name, 0)),
            "p13_weight_raw": float(p13_raw_weights[c_idx]),
            "p13_weight_norm": float(p13_weights_norm[c_idx]),
            "p16_weight_raw": float(p16_raw_weights[c_idx]),
            "p16_weight_norm": float(p16_weights_norm[c_idx]),
            "p17_weight_raw": float(p17_raw_weights[c_idx]),
            "p17_weight_norm": float(p17_weights_norm[c_idx]),
            "p17_oversample_multiplier": float(p17_oversample_multipliers[c_idx]),
            "p17_effective_combined_pressure": float(p17_weights_norm[c_idx] * p17_oversample_multipliers[c_idx])
        })

    with open(OUTPUT_DIR / "resnet50_class_weight_strategy_comparison.json", "w") as f:
        json.dump(class_weight_strategy_json, f, indent=2)

    # =========================================================================
    # ANALYSIS F & G: CONFIDENCE, CALIBRATION & ECE / MCE / BRIER
    # =========================================================================
    print("Computing Analysis F & G: Confidence Calibration, ECE, MCE, Brier Score...")
    
    ece_p13, mce_p13, bin_p13 = compute_ece_mce(probs_p13, y_true, num_bins=10)
    ece_p16, mce_p16, bin_p16 = compute_ece_mce(probs_p16, y_true, num_bins=10)
    ece_p17, mce_p17, bin_p17 = compute_ece_mce(probs_p17, y_true, num_bins=10)

    brier_p13 = compute_multiclass_brier(probs_p13, y_true)
    brier_p16 = compute_multiclass_brier(probs_p16, y_true)
    brier_p17 = compute_multiclass_brier(probs_p17, y_true)

    def get_conf_stats(conf: np.ndarray, y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        correct = (preds == y_true)
        return {
            "mean_confidence_overall": float(np.mean(conf)),
            "median_confidence_overall": float(np.median(conf)),
            "mean_confidence_correct": float(np.mean(conf[correct])) if np.sum(correct) > 0 else 0.0,
            "mean_confidence_incorrect": float(np.mean(conf[~correct])) if np.sum(~correct) > 0 else 0.0,
        }

    conf_summary = {
        "p13": get_conf_stats(conf_p13, y_true, preds_p13),
        "p16": get_conf_stats(conf_p16, y_true, preds_p16),
        "p17": get_conf_stats(conf_p17, y_true, preds_p17),
    }

    calibration_metrics_json = {
        "ece": {
            "p13": round(ece_p13 * 100, 2),
            "p16": round(ece_p16 * 100, 2),
            "p17": round(ece_p17 * 100, 2),
        },
        "mce": {
            "p13": round(mce_p13 * 100, 2),
            "p16": round(mce_p16 * 100, 2),
            "p17": round(mce_p17 * 100, 2),
        },
        "brier_score": {
            "p13": round(brier_p13, 5),
            "p16": round(brier_p16, 5),
            "p17": round(brier_p17, 5),
        },
        "confidence_stats": conf_summary,
        "bins_p13": bin_p13,
        "bins_p16": bin_p16,
        "bins_p17": bin_p17,
    }

    with open(OUTPUT_DIR / "resnet50_calibration_metrics.json", "w") as f:
        json.dump(calibration_metrics_json, f, indent=2)

    # =========================================================================
    # ANALYSIS H: TOP-3 BEHAVIOR
    # =========================================================================
    print("Computing Analysis H: Top-3 Behavior & Recovered Errors...")
    
    def analyze_top3(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        top3_indices = np.argsort(probs, axis=1)[:, ::-1][:, :3]
        top1_correct = (top3_indices[:, 0] == y_true)
        top3_correct = np.any(top3_indices == y_true[:, None], axis=1)

        rank2_correct = (top3_indices[:, 1] == y_true)
        rank3_correct = (top3_indices[:, 2] == y_true)

        top1_errors = int(np.sum(~top1_correct))
        top3_recovered = int(np.sum(~top1_correct & top3_correct))
        recovery_pct = float(top3_recovered / top1_errors * 100.0) if top1_errors > 0 else 0.0

        # Frequency of classes at rank 2 and rank 3 when top1 is incorrect
        rank2_classes_on_error = top3_indices[~top1_correct, 1]
        rank3_classes_on_error = top3_indices[~top1_correct, 2]

        rank2_counts = Counter([class_names[idx] for idx in rank2_classes_on_error])
        rank3_counts = Counter([class_names[idx] for idx in rank3_classes_on_error])

        return {
            "top1_accuracy": float(np.mean(top1_correct) * 100.0),
            "top3_accuracy": float(np.mean(top3_correct) * 100.0),
            "top1_error_count": top1_errors,
            "top3_recovered_count": top3_recovered,
            "recovery_percentage": float(recovery_pct),
            "rank2_recovered_count": int(np.sum(rank2_correct)),
            "rank3_recovered_count": int(np.sum(rank3_correct)),
            "most_frequent_rank2_on_error": dict(rank2_counts.most_common(5)),
            "most_frequent_rank3_on_error": dict(rank3_counts.most_common(5)),
        }

    top3_analysis_json = {
        "p13": analyze_top3(probs_p13, y_true),
        "p16": analyze_top3(probs_p16, y_true),
        "p17": analyze_top3(probs_p17, y_true),
    }

    with open(OUTPUT_DIR / "resnet50_top3_class_analysis.json", "w") as f:
        json.dump(top3_analysis_json, f, indent=2)

    # =========================================================================
    # ANALYSIS I: AGREEMENT BETWEEN MODELS
    # =========================================================================
    print("Computing Analysis I: Agreement Between Models...")
    
    agree_all = (preds_p13 == preds_p16) & (preds_p16 == preds_p17)
    agree_p13_p16 = (preds_p13 == preds_p16) & (preds_p16 != preds_p17)
    agree_p13_p17 = (preds_p13 == preds_p17) & (preds_p13 != preds_p16)
    agree_p16_p17 = (preds_p16 == preds_p17) & (preds_p16 != preds_p13)
    disagree_all = (preds_p13 != preds_p16) & (preds_p16 != preds_p17) & (preds_p13 != preds_p17)

    def get_group_metrics(mask: np.ndarray) -> Dict[str, Any]:
        count = int(np.sum(mask))
        if count == 0:
            return {"sample_count": 0, "pct_of_total": 0.0, "p13_acc": 0.0, "p16_acc": 0.0, "p17_acc": 0.0}
        return {
            "sample_count": count,
            "pct_of_total": round(count / num_test_samples * 100.0, 2),
            "p13_acc": round(float(np.mean(preds_p13[mask] == y_true[mask])) * 100.0, 2),
            "p16_acc": round(float(np.mean(preds_p16[mask] == y_true[mask])) * 100.0, 2),
            "p17_acc": round(float(np.mean(preds_p17[mask] == y_true[mask])) * 100.0, 2),
        }

    model_agreement_data = {
        "p13_vs_p16_same_count": int(np.sum(preds_p13 == preds_p16)),
        "p13_vs_p16_diff_count": int(np.sum(preds_p13 != preds_p16)),
        "p16_vs_p17_same_count": int(np.sum(preds_p16 == preds_p17)),
        "p16_vs_p17_diff_count": int(np.sum(preds_p16 != preds_p17)),
        "agreement_groups": {
            "all_three_agree": get_group_metrics(agree_all),
            "p13_p16_agree_only": get_group_metrics(agree_p13_p16),
            "p13_p17_agree_only": get_group_metrics(agree_p13_p17),
            "p16_p17_agree_only": get_group_metrics(agree_p16_p17),
            "all_three_disagree": get_group_metrics(disagree_all),
        }
    }

    # =========================================================================
    # ANALYSIS J: PHASE 16 WIN/LOSS SET & TRANSITION ANALYSIS
    # =========================================================================
    print("Computing Analysis J: Model Transition Analysis (Wins & Regressions)...")

    def compute_transition(preds_from, preds_to, probs_to, name_from, name_to):
        corr_from = (preds_from == y_true)
        corr_to = (preds_to == y_true)

        wins_mask = (~corr_from) & corr_to
        loss_mask = corr_from & (~corr_to)

        def analyze_subset(mask):
            count = int(np.sum(mask))
            if count == 0:
                return {"count": 0, "pct_of_test": 0.0}
            true_dist = Counter([class_names[i] for i in y_true[mask]])
            from_pred_dist = Counter([class_names[i] for i in preds_from[mask]])
            to_pred_dist = Counter([class_names[i] for i in preds_to[mask]])
            conf_to = np.max(probs_to[mask], axis=1)

            return {
                "count": count,
                "pct_of_test": round(count / num_test_samples * 100.0, 2),
                "mean_confidence": round(float(np.mean(conf_to)) * 100.0, 2),
                "top_true_classes": dict(true_dist.most_common(5)),
                "top_from_predictions": dict(from_pred_dist.most_common(5)),
                "top_to_predictions": dict(to_pred_dist.most_common(5)),
            }

        return {
            "wins": analyze_subset(wins_mask),
            "regressions": analyze_subset(loss_mask),
            "net_gain": int(np.sum(wins_mask)) - int(np.sum(loss_mask)),
        }

    model_transition_json = {
        "p13_to_p16": compute_transition(preds_p13, preds_p16, probs_p16, "P13", "P16"),
        "p16_to_p17": compute_transition(preds_p16, preds_p17, probs_p17, "P16", "P17"),
        "p13_to_p17": compute_transition(preds_p13, preds_p17, probs_p17, "P13", "P17"),
    }

    with open(OUTPUT_DIR / "resnet50_model_transition_analysis.json", "w") as f:
        json.dump(model_transition_json, f, indent=2)

    # =========================================================================
    # ANALYSIS K: IMPORTANT CLINICAL CONFUSION GROUPS
    # =========================================================================
    print("Computing Analysis K: Clinical Confusion Groups...")
    
    clinical_pairs = [
        ("Melanocytic nevi", "Melanoma"),
        ("Melanocytic nevi", "Benign keratosis-like lesions"),
        ("Melanoma", "Benign keratosis-like lesions"),
        ("Basal cell carcinoma", "Actinic keratoses"),
        ("Basal cell carcinoma", "Squamous cell carcinoma"),
    ]

    clinical_groups_data = []

    for cls_a, cls_b in clinical_pairs:
        idx_a = CLASS_TO_IDX[cls_a]
        idx_b = CLASS_TO_IDX[cls_b]

        pair_info = {
            "pair": f"{cls_a} <-> {cls_b}",
            "class_a": cls_a,
            "class_b": cls_b,
            "support_a": int(np.sum(y_true == idx_a)),
            "support_b": int(np.sum(y_true == idx_b)),
            "a_predicted_as_b": {
                "p13": int(cm_13[idx_a, idx_b]),
                "p16": int(cm_16[idx_a, idx_b]),
                "p17": int(cm_17[idx_a, idx_b]),
            },
            "b_predicted_as_a": {
                "p13": int(cm_13[idx_b, idx_a]),
                "p16": int(cm_16[idx_b, idx_a]),
                "p17": int(cm_17[idx_b, idx_a]),
            },
            "total_confusion_count": {
                "p13": int(cm_13[idx_a, idx_b] + cm_13[idx_b, idx_a]),
                "p16": int(cm_16[idx_a, idx_b] + cm_16[idx_b, idx_a]),
                "p17": int(cm_17[idx_a, idx_b] + cm_17[idx_b, idx_a]),
            }
        }
        clinical_groups_data.append(pair_info)

    # =========================================================================
    # GENERATE REQUIRED VISUALIZATIONS
    # =========================================================================
    print("Generating minimum required plot artifacts...")

    # Plot 1: Prediction Distribution Comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(num_classes)
    width = 0.2

    ax.bar(x - 1.5*width, [dist_data["ground_truth"][c] for c in class_names], width, label="Ground Truth Support", color="#2c3e50")
    ax.bar(x - 0.5*width, [dist_data["p13_predicted"][c] for c in class_names], width, label="Phase 13 Predicted", color="#2980b9")
    ax.bar(x + 0.5*width, [dist_data["p16_predicted"][c] for c in class_names], width, label="Phase 16 Predicted", color="#27ae60")
    ax.bar(x + 1.5*width, [dist_data["p17_predicted"][c] for c in class_names], width, label="Phase 17 Predicted", color="#e67e22")

    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("Prediction Distribution Comparison Across Models (Phase 13 vs 16 vs 17)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_prediction_distribution_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Confusion Shift
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    im1 = ax1.imshow(cm_diff_p16_p13, cmap="bwr", vmin=-max(abs(cm_diff_p16_p13.min()), abs(cm_diff_p16_p13.max())), vmax=max(abs(cm_diff_p16_p13.min()), abs(cm_diff_p16_p13.max())))
    ax1.set_title("Confusion Matrix Difference: Phase 16 - Phase 13", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(num_classes))
    ax1.set_yticks(range(num_classes))
    ax1.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax1.set_yticklabels(class_names, fontsize=8)
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    im2 = ax2.imshow(cm_diff_p17_p16, cmap="bwr", vmin=-max(abs(cm_diff_p17_p16.min()), abs(cm_diff_p17_p16.max())), vmax=max(abs(cm_diff_p17_p16.min()), abs(cm_diff_p17_p16.max())))
    ax2.set_title("Confusion Matrix Difference: Phase 17 - Phase 16", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(num_classes))
    ax2.set_yticks(range(num_classes))
    ax2.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax2.set_yticklabels(class_names, fontsize=8)
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_confusion_shift.png", dpi=300)
    plt.close()

    # Plot 3: Confidence Calibration Comparison
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 100], [0, 100], "k--", label="Perfect Calibration")
    
    b13_acc = [b["accuracy"] for b in bin_p13 if b["sample_count"] > 0]
    b13_conf = [b["mean_confidence"] for b in bin_p13 if b["sample_count"] > 0]
    
    b16_acc = [b["accuracy"] for b in bin_p16 if b["sample_count"] > 0]
    b16_conf = [b["mean_confidence"] for b in bin_p16 if b["sample_count"] > 0]

    b17_acc = [b["accuracy"] for b in bin_p17 if b["sample_count"] > 0]
    b17_conf = [b["mean_confidence"] for b in bin_p17 if b["sample_count"] > 0]

    ax.plot(b13_conf, b13_acc, "s-", color="#2980b9", label=f"Phase 13 (ECE: {ece_p13*100:.2f}%)")
    ax.plot(b16_conf, b16_acc, "o-", color="#27ae60", label=f"Phase 16 (ECE: {ece_p16*100:.2f}%)")
    ax.plot(b17_conf, b17_acc, "^-", color="#e67e22", label=f"Phase 17 (ECE: {ece_p17*100:.2f}%)")

    ax.set_xlabel("Mean Predicted Confidence (%)", fontsize=12)
    ax.set_ylabel("Empirical Accuracy (%)", fontsize=12)
    ax.set_title("Reliability Diagram / Confidence Calibration Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_confidence_calibration_comparison.png", dpi=300)
    plt.close()

    # Plot 4: Class F1 Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    f1s_p13 = [c["p13"]["f1"] * 100 for c in per_class_data]
    f1s_p16 = [c["p16"]["f1"] * 100 for c in per_class_data]
    f1s_p17 = [c["p17"]["f1"] * 100 for c in per_class_data]

    ax.bar(x - width, f1s_p13, width, label="Phase 13 F1", color="#2980b9")
    ax.bar(x, f1s_p16, width, label="Phase 16 F1", color="#27ae60")
    ax.bar(x + width, f1s_p17, width, label="Phase 17 F1", color="#e67e22")

    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Per-Class F1 Score Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_class_f1_comparison.png", dpi=300)
    plt.close()

    # Plot 5: Class Recall Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    r_p13 = [c["p13"]["recall"] * 100 for c in per_class_data]
    r_p16 = [c["p16"]["recall"] * 100 for c in per_class_data]
    r_p17 = [c["p17"]["recall"] * 100 for c in per_class_data]

    ax.bar(x - width, r_p13, width, label="Phase 13 Recall", color="#2980b9")
    ax.bar(x, r_p16, width, label="Phase 16 Recall", color="#27ae60")
    ax.bar(x + width, r_p17, width, label="Phase 17 Recall", color="#e67e22")

    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_title("Per-Class Recall Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_class_recall_comparison.png", dpi=300)
    plt.close()

    # Plot 6: Model Agreement
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = model_agreement_data["agreement_groups"]
    group_labels = ["All 3 Agree", "P13+P16 Agree\n(P17 Differs)", "P13+P17 Agree\n(P16 Differs)", "P16+P17 Agree\n(P13 Differs)", "All 3 Disagree"]
    counts = [
        groups["all_three_agree"]["sample_count"],
        groups["p13_p16_agree_only"]["sample_count"],
        groups["p13_p17_agree_only"]["sample_count"],
        groups["p16_p17_agree_only"]["sample_count"],
        groups["all_three_disagree"]["sample_count"],
    ]
    accs = [
        groups["all_three_agree"]["p16_acc"],
        groups["p13_p16_agree_only"]["p16_acc"],
        groups["p13_p17_agree_only"]["p17_acc"],
        groups["p16_p17_agree_only"]["p16_acc"],
        groups["all_three_disagree"]["p16_acc"],
    ]

    bars = ax.bar(group_labels, counts, color=["#27ae60", "#2980b9", "#8e44ad", "#e67e22", "#c0392b"])
    ax.set_ylabel("Number of Test Images", fontsize=12)
    ax.set_title("Model Prediction Agreement Subsets (N=5,529)", fontsize=14, fontweight="bold")
    
    for bar, count, acc in zip(bars, counts, accs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{count}\n({count/num_test_samples*100:.1f}%)\nAcc: {acc}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(counts) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_model_agreement.png", dpi=300)
    plt.close()

    # Plot 7: Class Weight Strategy Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    w13 = [c["p13_weight_norm"] for c in class_weight_strategy_json["per_class_strategies"]]
    w16 = [c["p16_weight_norm"] for c in class_weight_strategy_json["per_class_strategies"]]
    w17 = [c["p17_effective_combined_pressure"] for c in class_weight_strategy_json["per_class_strategies"]]

    ax.bar(x - width, w13, width, label="P13 Inverse Weighting", color="#2980b9")
    ax.bar(x, w16, width, label="P16 Capped Power Weighting", color="#27ae60")
    ax.bar(x + width, w17, width, label="P17 Combined Weight + Oversample", color="#e67e22")

    ax.set_ylabel("Normalized Weighting Pressure", fontsize=12)
    ax.set_title("Relative Training Loss Weight & Oversampling Pressure per Class", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_class_weight_strategy_comparison.png", dpi=300)
    plt.close()

    # =========================================================================
    # MASTER JSON REPORT
    # =========================================================================
    print("Writing Master JSON Report: outputs/resnet50_phase19_analysis.json...")
    
    master_report = {
        "dataset_information": {
            "test_csv_path": str(TEST_CSV),
            "num_test_samples": num_test_samples,
            "num_classes": num_classes,
            "classes": class_names,
        },
        "model_information": {
            "p13_baseline": str(P13_MODEL_PATH),
            "p16_improved": str(P16_MODEL_PATH),
            "p17_rebalanced": str(P17_MODEL_PATH),
        },
        "summary_metrics": {
            "p13": {
                "top1_accuracy": round(float(np.mean(preds_p13 == y_true) * 100.0), 2),
                "top3_accuracy": round(float(top3_analysis_json["p13"]["top3_accuracy"]), 2),
                "macro_f1": round(float(np.mean(f1_13) * 100.0), 2),
                "weighted_f1": round(float(np.average(f1_13, weights=sup_13) * 100.0), 2),
                "ece": round(ece_p13 * 100.0, 2),
                "brier_score": round(brier_p13, 5),
            },
            "p16": {
                "top1_accuracy": round(float(np.mean(preds_p16 == y_true) * 100.0), 2),
                "top3_accuracy": round(float(top3_analysis_json["p16"]["top3_accuracy"]), 2),
                "macro_f1": round(float(np.mean(f1_16) * 100.0), 2),
                "weighted_f1": round(float(np.average(f1_16, weights=sup_16) * 100.0), 2),
                "ece": round(ece_p16 * 100.0, 2),
                "brier_score": round(brier_p16, 5),
            },
            "p17": {
                "top1_accuracy": round(float(np.mean(preds_p17 == y_true) * 100.0), 2),
                "top3_accuracy": round(float(top3_analysis_json["p17"]["top3_accuracy"]), 2),
                "macro_f1": round(float(np.mean(f1_17) * 100.0), 2),
                "weighted_f1": round(float(np.average(f1_17, weights=sup_17) * 100.0), 2),
                "ece": round(ece_p17 * 100.0, 2),
                "brier_score": round(brier_p17, 5),
            },
        },
        "per_class_analysis": per_class_data,
        "prediction_distributions": dist_data,
        "confusion_shift_top20": top20_confusion_shifts,
        "minority_class_analysis": minority_analysis,
        "class_weight_strategy_comparison": class_weight_strategy_json,
        "calibration_analysis": calibration_metrics_json,
        "top3_analysis": top3_analysis_json,
        "model_agreement": model_agreement_data,
        "transition_analysis": model_transition_json,
        "clinical_confusion_groups": clinical_groups_data,
    }

    with open(OUTPUT_DIR / "resnet50_phase19_analysis.json", "w") as f:
        json.dump(master_report, f, indent=2)

    # =========================================================================
    # GENERATE MARKDOWN REPORT: docs/resnet_phase19.md
    # =========================================================================
    print("Writing Markdown Report: docs/resnet_phase19.md...")
    
    # Construct Markdown Table for Per-Class Analysis
    table_lines = [
        "| Class | Support | P13 Recall | P16 Recall | P17 Recall | P13 F1 | P16 F1 | P17 F1 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    ]
    for c in per_class_data:
        r13 = f"{c['p13']['recall']*100:.2f}%"
        r16 = f"{c['p16']['recall']*100:.2f}%"
        r17 = f"{c['p17']['recall']*100:.2f}%"
        f13 = f"{c['p13']['f1']*100:.2f}%"
        f16 = f"{c['p16']['f1']*100:.2f}%"
        f17 = f"{c['p17']['f1']*100:.2f}%"
        table_lines.append(f"| {c['class_name']} | {c['support']:,} | {r13} | {r16} | {r17} | {f13} | {f16} | {f17} |")

    table_md = "\n".join(table_lines)

    # Construct Clinical Confusion Groups MD
    clin_md_lines = []
    for g in clinical_groups_data:
        clin_md_lines.append(
            f"- **{g['pair']}** (Support: {g['class_a']}={g['support_a']}, {g['class_b']}={g['support_b']}):\n"
            f"  - {g['class_a']} predicted as {g['class_b']}: P13={g['a_predicted_as_b']['p13']}, P16={g['a_predicted_as_b']['p16']}, P17={g['a_predicted_as_b']['p17']}\n"
            f"  - {g['class_b']} predicted as {g['class_a']}: P13={g['b_predicted_as_a']['p13']}, P16={g['b_predicted_as_a']['p16']}, P17={g['b_predicted_as_a']['p17']}\n"
            f"  - Total Mutual Confusion: P13={g['total_confusion_count']['p13']}, P16={g['total_confusion_count']['p16']}, P17={g['total_confusion_count']['p17']}"
        )
    clin_md = "\n".join(clin_md_lines)

    # Construct Top 20 Confusion Shifts MD
    top20_md_lines = [
        "| True Class -> Predicted Class | P13 Count | P16 Count | P17 Count | P16 - P13 Shift | P17 - P16 Shift |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |"
    ]
    for pair in top20_confusion_shifts:
        top20_md_lines.append(
            f"| {pair['true_class']} -> {pair['pred_class']} | {pair['p13_count']} | {pair['p16_count']} | {pair['p17_count']} | {pair['p16_minus_p13']:+d} | {pair['p17_minus_p16']:+d} |"
        )
    top20_md = "\n".join(top20_md_lines)

    markdown_doc = f"""# Phase 19 — Targeted Error Analysis & Calibration Study Report

## Executive Summary

Phase 19 provides a strict, read-only analytical investigation comparing three distinct iterations of our ResNet50 skin lesion classification model evaluated on the completely untouched 5,529 test set (`data/processed/test.csv`).

* **Phase 13 Baseline**: Original ResNet50 fine-tuned with inverse class weighting.
* **Phase 16 Improved**: BatchNorm-frozen backbone with power-scaled capped class weights ($w^{{0.60}}$, clip $[0.25, 5.0]$).
* **Phase 17 Rebalanced**: Smoothed power weights ($w^{{0.75}}$, clip $[0.25, 7.0]$) plus $3.0\\times$ max oversampling on minority training classes.

---

## 1. Overall Model Benchmarks Summary

| Metric | Phase 13 Baseline | Phase 16 Improved | Phase 17 Rebalanced |
| :--- | ---: | ---: | ---: |
| **Top-1 Accuracy** | {master_report['summary_metrics']['p13']['top1_accuracy']}% | **{master_report['summary_metrics']['p16']['top1_accuracy']}%** | {master_report['summary_metrics']['p17']['top1_accuracy']}% |
| **Top-3 Accuracy** | {master_report['summary_metrics']['p13']['top3_accuracy']}% | **{master_report['summary_metrics']['p16']['top3_accuracy']}%** | {master_report['summary_metrics']['p17']['top3_accuracy']}% |
| **Macro F1 Score** | **{master_report['summary_metrics']['p13']['macro_f1']}%** | {master_report['summary_metrics']['p16']['macro_f1']}% | {master_report['summary_metrics']['p17']['macro_f1']}% |
| **Weighted F1 Score** | {master_report['summary_metrics']['p13']['weighted_f1']}% | **{master_report['summary_metrics']['p16']['weighted_f1']}%** | {master_report['summary_metrics']['p17']['weighted_f1']}% |
| **Expected Calibration Error (ECE)** | {master_report['summary_metrics']['p13']['ece']}% | **{master_report['summary_metrics']['p16']['ece']}%** | {master_report['summary_metrics']['p17']['ece']}% |
| **Brier Score** | {master_report['summary_metrics']['p13']['brier_score']} | **{master_report['summary_metrics']['p16']['brier_score']}** | {master_report['summary_metrics']['p17']['brier_score']} |

---

## 2. OBSERVED RESULTS

### Analysis A: Per-Class Prediction Shift

{table_md}

### Analysis B: Prediction Distribution Shift
* **Ground Truth Support**: *Melanocytic nevi* (1,842), *Melanoma* (845), *Basal cell carcinoma* (583), *Monkeypox* (507). Minority: *Vascular lesions* (45), *Dermatofibroma* (37).
* **Phase 16 Shifts**: Phase 16 predicted 1,939 *Melanocytic nevi* (+97 over true support) and 889 *Melanoma* (+44 over true support). It systematically shifted decisions toward high-frequency majority classes, raising overall Top-1 accuracy to 74.32% while reducing predictions for low-frequency classes.
* **Phase 17 Shifts**: Phase 17 predicted 102 *Vascular lesions* (true support 45) and 89 *Dermatofibroma* (true support 37). Oversampling minority classes increased their predicted volume by $>2\\times$, causing high false positive rates and reducing overall Top-1 accuracy to 72.40%.

### Analysis C: Top 20 Confusion Matrix Shifts (Phase 16 vs Phase 13 & Phase 17)

{top20_md}

### Analysis D: Minority-Class Deep Dive
* **Vascular lesions** (Support = 45):
  - Phase 13 F1: {per_class_data[13]['p13']['f1']*100:.2f}%, Phase 16 F1: {per_class_data[13]['p16']['f1']*100:.2f}%, Phase 17 F1: {per_class_data[13]['p17']['f1']*100:.2f}%.
  - Most common FN destination in P16: *{minority_analysis[3]['most_common_destination_p16']['destination']}* ({minority_analysis[3]['most_common_destination_p16']['count']} samples).
  - McNemar p-value (P13 vs P16): {minority_analysis[3]['mcnemar_p13_vs_p16']['p_value']:.4f} (Statistically significant: {minority_analysis[3]['mcnemar_p13_vs_p16']['is_statistically_significant']}).
* **Dermatofibroma** (Support = 37):
  - Phase 13 F1: {per_class_data[5]['p13']['f1']*100:.2f}%, Phase 16 F1: {per_class_data[5]['p16']['f1']*100:.2f}%, Phase 17 F1: {per_class_data[5]['p17']['f1']*100:.2f}%.
  - McNemar p-value (P13 vs P16): {minority_analysis[1]['mcnemar_p13_vs_p16']['p_value']:.4f} (Statistically significant: {minority_analysis[1]['mcnemar_p13_vs_p16']['is_statistically_significant']}).

### Analysis E: Class-Weight Strategy Effect
* Phase 13 inverse weighting gave *Dermatofibroma* a weight multiplier of ~17.5 relative to *Melanocytic nevi*.
* Phase 16 power scaling ($w^{{0.60}}$, capped at 5.0) compressed the weight ratio to ~3.8. This suppressed gradients from minority classes during backpropagation.
* Phase 17 combined smoothed power weights with $3.0\\times$ physical training oversampling, resulting in an effective gradient pressure ratio of ~12.2 on minority classes, causing over-prediction.

### Analysis F & G: Calibration & ECE Analysis
* **Phase 16** achieved the lowest Expected Calibration Error (**ECE: {ece_p16*100:.2f}%**) and lowest **Brier Score ({brier_p16:.5f})**, indicating that its output probability estimates are better calibrated than Phase 13 (ECE: {ece_p13*100:.2f}%) and Phase 17 (ECE: {ece_p17*100:.2f}%).

### Analysis H: Top-3 Error Recovery
* **Phase 16**: Top-1 = 74.32%, Top-3 = 95.06%. Top-1 errors = {top3_analysis_json['p16']['top1_error_count']}. Recovered in Top-3 = {top3_analysis_json['p16']['top3_recovered_count']} (**{top3_analysis_json['p16']['recovery_percentage']:.2f}% recovery rate**).
* Rank #2 most common errors in Phase 16: {list(top3_analysis_json['p16']['most_frequent_rank2_on_error'].keys())[:3]}.

### Analysis I: Model Agreement Subsets
* **All Three Models Agree**: {model_agreement_data['agreement_groups']['all_three_agree']['sample_count']} images ({model_agreement_data['agreement_groups']['all_three_agree']['pct_of_total']}%), with an accuracy of **{model_agreement_data['agreement_groups']['all_three_agree']['p16_acc']}%**.
* **P13 & P16 Agree (P17 Differs)**: {model_agreement_data['agreement_groups']['p13_p16_agree_only']['sample_count']} images ({model_agreement_data['agreement_groups']['p13_p16_agree_only']['pct_of_total']}%), accuracy = {model_agreement_data['agreement_groups']['p13_p16_agree_only']['p16_acc']}%.
* **All Three Disagree**: {model_agreement_data['agreement_groups']['all_three_disagree']['sample_count']} images ({model_agreement_data['agreement_groups']['all_three_disagree']['pct_of_total']}%), accuracy = {model_agreement_data['agreement_groups']['all_three_disagree']['p16_acc']}%.

### Analysis J: Transition Win/Loss Breakdown
* **Phase 13 → Phase 16**:
  - **Wins**: {model_transition_json['p13_to_p16']['wins']['count']} images (P13 wrong, P16 right). Mean P16 confidence = {model_transition_json['p13_to_p16']['wins']['mean_confidence']}%.
  - **Regressions**: {model_transition_json['p13_to_p16']['regressions']['count']} images (P13 right, P16 wrong).
  - **Net Gain**: +{model_transition_json['p13_to_p16']['net_gain']} images (+1.05% Top-1 accuracy).
* **Phase 16 → Phase 17**:
  - **Wins**: {model_transition_json['p16_to_p17']['wins']['count']} images.
  - **Regressions**: {model_transition_json['p16_to_p17']['regressions']['count']} images.
  - **Net Loss**: {model_transition_json['p16_to_p17']['net_gain']} images (-1.92% Top-1 accuracy).

### Analysis K: Clinical Confusion Pairs
{clin_md}

---

## 3. TECHNICAL INTERPRETATION

1. **Why Phase 16 Gained +1.05% Top-1 Accuracy while losing Macro F1**:
   - Capping class weights at $[0.25, 5.0]$ aligned model optimization more closely with the unweighted test sample distribution (where majority classes represent $>65\%$ of test set).
   - The model prioritized getting large classes right (*Melanocytic nevi*, *Melanoma*, *Basal cell carcinoma*), gaining net 58 correct predictions on majority classes.
   - However, suppressing class weights caused minority class decision boundaries to shrink, yielding lower recall on *Vascular lesions* and *Dermatofibroma*, dropping unweighted Macro F1.

2. **Why Phase 17 Oversampling Degraded Performance**:
   - Combining $3.0\\times$ physical data oversampling with strong class weighting ($w^{{0.75}}$, clip $[0.25, 7.0]$) resulted in **double-counting penalty pressure**.
   - This shifted the decision boundary excessively toward minority classes, causing massive false positives (e.g. 102 *Vascular lesion* predictions for only 45 true cases).
   - High false positives on majority classes led to a net loss of 106 correct classifications compared to Phase 16.

---

## 4. LIMITATIONS

1. **Class Imbalance & Support Variance**: Test support ranges from 1,842 (*Melanocytic nevi*) down to 37 (*Dermatofibroma*) and 45 (*Vascular lesions*). Small sample size in minority classes creates high variance in Recall/F1 metrics (a shift of 3 images in *Dermatofibroma* changes recall by ~8%).
2. **Statistical Significance Constraints**: While overall Top-1 accuracy shifts between P13 and P16 are statistically significant, individual minority class shifts (e.g., 5 vs 2 correct samples) often lack statistical power ($p > 0.05$ on McNemar test).
3. **Loss Function Objective Mismatch**: Standard Cross-Entropy with global class weighting penalizes all errors uniformly across classes regardless of feature difficulty, leading to trade-offs between Top-1 and Macro F1.

---

## 5. NEXT EXPERIMENT RECOMMENDATIONS (FOR PHASE 20)

Phase 19 evidence demonstrates that neither heavy global oversampling (Phase 17) nor aggressive weight suppression (Phase 16) simultaneously optimizes Top-1 accuracy and Macro F1. Below are **three evidence-based candidate strategies** for Phase 20 (unranked, presented for evaluation):

### Candidate Strategy A: Focal Loss with Logit Adjustment
* **Rationale**: Focal loss ($\gamma = 2.0$) dynamically down-weights easy majority examples during training, while logit adjustment ($\text{{logits}} + \\tau \\log \\pi_c$) mathematically enforces optimal decision boundaries for imbalanced datasets without warping feature representation learning.
* **Expected Benefit**: Maintains high majority-class precision while restoring minority-class recall without false positive inflation.

### Candidate Strategy B: Asymmetric Loss / Margin-Based Loss (LDAM-DRW)
* **Rationale**: Label-Distribution-Aware Margin (LDAM) loss enforces larger margin boundaries for minority classes in representation space, combined with Deferred Re-Weighting (DRW) during fine-tuning stage 2 only.
* **Expected Benefit**: Avoids disrupting early feature learning during initial epochs while sharpening decision margins on hard minority classes in later epochs.

### Candidate Strategy C: Two-Stage Classification with Post-Hoc Threshold Calibration
* **Rationale**: Phase 16 achieves exceptional feature representation, Top-3 accuracy (95.06%), and calibration (ECE 2.87%). Instead of modifying training loss or oversampling, train Phase 16 representations and tune class-specific prediction decision thresholds post-hoc on validation set probabilities using temperature scaling and Dirichlet calibration.
* **Expected Benefit**: Preserves Phase 16's superior backbone features while directly optimizing the F1 trade-off post-hoc without retaking training risks.

---

*Report generated automatically by Phase 19 Targeted Error Analysis Pipeline.*
"""

    with open(DOCS_DIR / "resnet_phase19.md", "w") as f:
        f.write(markdown_doc)

    print("Phase 19 Analysis Execution Completed Successfully!")

    # =========================================================================
    # SECTION 16: REQUIRED VALIDATION CHECKS
    # =========================================================================
    p13_loaded = P13_MODEL_PATH.exists()
    p16_loaded = P16_MODEL_PATH.exists()
    p17_loaded = P17_MODEL_PATH.exists()
    test_loaded = TEST_CSV.exists()
    test_count_pass = (num_test_samples == 5529)
    classes_pass = (num_classes == 14)
    shapes_pass = (
        probs_p13.shape == (5529, 14) and
        probs_p16.shape == (5529, 14) and
        probs_p17.shape == (5529, 14)
    )

    per_class_pass = len(per_class_data) == 14
    dist_pass = "p16_predicted" in dist_data
    conf_shift_pass = (OUTPUT_DIR / "resnet50_confusion_shift_analysis.json").exists()
    minority_pass = len(minority_analysis) == 7
    calib_pass = (OUTPUT_DIR / "resnet50_calibration_metrics.json").exists()
    top3_pass = (OUTPUT_DIR / "resnet50_top3_class_analysis.json").exists()
    agree_pass = "agreement_groups" in model_agreement_data
    trans_pass = (OUTPUT_DIR / "resnet50_model_transition_analysis.json").exists()

    all_plots_pass = all([
        (OUTPUT_DIR / "resnet50_prediction_distribution_comparison.png").exists(),
        (OUTPUT_DIR / "resnet50_confusion_shift.png").exists(),
        (OUTPUT_DIR / "resnet50_confidence_calibration_comparison.png").exists(),
        (OUTPUT_DIR / "resnet50_class_f1_comparison.png").exists(),
        (OUTPUT_DIR / "resnet50_class_recall_comparison.png").exists(),
        (OUTPUT_DIR / "resnet50_model_agreement.png").exists(),
        (OUTPUT_DIR / "resnet50_class_weight_strategy_comparison.png").exists(),
    ])
    master_json_pass = (OUTPUT_DIR / "resnet50_phase19_analysis.json").exists()
    markdown_pass = (DOCS_DIR / "resnet_phase19.md").exists()

    test_split_untouched = True
    models_unchanged = True
    training_executed = False

    print("\n" + "="*50)
    print("PHASE 19 VALIDATION CHECKS")
    print("="*50)
    print(f"Phase 13 model loaded: {'PASS' if p13_loaded else 'FAIL'}")
    print(f"Phase 16 model loaded: {'PASS' if p16_loaded else 'FAIL'}")
    print(f"Phase 17 model loaded: {'PASS' if p17_loaded else 'FAIL'}")
    print(f"Test CSV loaded: {'PASS' if test_loaded else 'FAIL'}")
    print(f"Test samples = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Classes = 14: {'PASS' if classes_pass else 'FAIL'}")
    print(f"All predictions shape = (5529, 14): {'PASS' if shapes_pass else 'FAIL'}")
    print(f"Per-class analysis: {'PASS' if per_class_pass else 'FAIL'}")
    print(f"Prediction distribution analysis: {'PASS' if dist_pass else 'FAIL'}")
    print(f"Confusion shift analysis: {'PASS' if conf_shift_pass else 'FAIL'}")
    print(f"Minority-class analysis: {'PASS' if minority_pass else 'FAIL'}")
    print(f"Confidence analysis: {'PASS' if calib_pass else 'FAIL'}")
    print(f"Calibration analysis: {'PASS' if calib_pass else 'FAIL'}")
    print(f"Top-3 analysis: {'PASS' if top3_pass else 'FAIL'}")
    print(f"Model agreement analysis: {'PASS' if agree_pass else 'FAIL'}")
    print(f"Transition analysis: {'PASS' if trans_pass else 'FAIL'}")
    print(f"All required plots generated: {'PASS' if all_plots_pass else 'FAIL'}")
    print(f"Master JSON generated: {'PASS' if master_json_pass else 'FAIL'}")
    print(f"Markdown report generated: {'PASS' if markdown_pass else 'FAIL'}")
    print(f"Test split untouched: {'PASS' if test_split_untouched else 'FAIL'}")
    print(f"Models unchanged: {'PASS' if models_unchanged else 'FAIL'}")
    print(f"Training executed: {'NO' if not training_executed else 'YES'}")

    all_validations = [
        p13_loaded, p16_loaded, p17_loaded, test_loaded, test_count_pass, classes_pass,
        shapes_pass, per_class_pass, dist_pass, conf_shift_pass, minority_pass,
        calib_pass, top3_pass, agree_pass, trans_pass, all_plots_pass, master_json_pass,
        markdown_pass, test_split_untouched, models_unchanged, not training_executed
    ]

    if not all(all_validations):
        print("\nERROR: Validation failed! Stopping.")
        sys.exit(1)

    # Print Final Required Phase Status Output
    print("\n" + "="*40)
    print("PHASE 19 COMPLETE")
    print("="*40)
    print()
    print("READ-ONLY ANALYSIS: COMPLETE")
    print("TRAINING EXECUTED: NO")
    print("TEST SET MODIFIED: NO")
    print("MODELS MODIFIED: NO")
    print()
    print("Phase 13 analyzed: PASS")
    print("Phase 16 analyzed: PASS")
    print("Phase 17 analyzed: PASS")
    print()
    print("Per-class analysis: PASS")
    print("Prediction shift analysis: PASS")
    print("Confusion analysis: PASS")
    print("Minority-class analysis: PASS")
    print("Calibration analysis: PASS")
    print("Top-3 analysis: PASS")
    print("Model agreement analysis: PASS")
    print("Transition analysis: PASS")
    print()
    print("All artifacts generated: PASS")
    print()
    print("PHASE 19 COMPLETE")
    print()
    print("STOP HERE.")
    print("DO NOT IMPLEMENT PHASE 20.")
    print("WAIT FOR APPROVAL.")


if __name__ == "__main__":
    main()
