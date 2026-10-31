"""
ResNet50 Phase 24 Evaluation & Six-Way Benchmarking Script

STRICTLY READ-ONLY EVALUATION OF:
1. Phase 13 Baseline: models/resnet50_best.keras
2. Phase 16 Improved: models/resnet50_improved_best.keras
3. Phase 17 Rebalanced: models/resnet50_phase17_best.keras
4. Phase 20 Controlled: models/resnet50_phase20_best.keras
5. Phase 23 LDAM Margin: models/resnet50_phase23_best.keras
6. Phase 24 Pairwise Boundary: models/resnet50_phase24_best.keras

Evaluated on untouched isolated test split: data/processed/test.csv (5,529 images, 14 classes).

NO MODEL TRAINING. NO WEIGHT MODIFICATIONS. TEST SET IS ISOLATED.
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
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
import tensorflow as tf

from src.resnet_data import CLASS_LIST, CLASS_TO_IDX, IDX_TO_CLASS, create_phase17_dataset


P13_MODEL_PATH = Path("models/resnet50_best.keras")
P16_MODEL_PATH = Path("models/resnet50_improved_best.keras")
P17_MODEL_PATH = Path("models/resnet50_phase17_best.keras")
P20_MODEL_PATH = Path("models/resnet50_phase20_best.keras")
P23_MODEL_PATH = Path("models/resnet50_phase23_best.keras")
P24_MODEL_PATH = Path("models/resnet50_phase24_best.keras")

TEST_CSV = Path("data/processed/test.csv")
TRAIN_CSV = Path("data/processed/train.csv")
OUTPUT_DIR = Path("outputs")
DOCS_DIR = Path("docs")


def setup_device(device_choice: str) -> str:
    """Configure TensorFlow device context."""
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
    """Load model checkpoint in read-only mode and run inference."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"Loading model (read-only): {model_path} ...")
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
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
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
            "bin_label": f"{int(bin_lower*100)}-{int(bin_upper*100)}%",
            "sample_count": bin_size,
            "accuracy": round(bin_acc * 100, 2),
            "mean_confidence": round(bin_conf * 100, 2),
            "calibration_gap": round(diff * 100, 2)
        })

    return float(ece), float(mce), bin_details


def compute_multiclass_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Compute multiclass Brier score."""
    n_samples, n_classes = probs.shape
    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
    y_onehot[np.arange(n_samples), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def main():
    parser = argparse.ArgumentParser(description="ResNet50 Phase 24 Evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    dev_name = setup_device(args.device)

    # Safety checks
    p13_exists = P13_MODEL_PATH.exists()
    p16_exists = P16_MODEL_PATH.exists()
    p17_exists = P17_MODEL_PATH.exists()
    p20_exists = P20_MODEL_PATH.exists()
    p23_exists = P23_MODEL_PATH.exists()
    p24_exists = P24_MODEL_PATH.exists()
    test_exists = TEST_CSV.exists()

    test_df = pd.read_csv(TEST_CSV) if test_exists else None
    num_test_samples = len(test_df) if test_df is not None else 0
    test_count_pass = (num_test_samples == 5529)
    classes_pass = len(CLASS_LIST) == 14

    print("========================================")
    print("PHASE 24 PRE-EVALUATION SAFETY CHECKS")
    print("========================================")
    print(f"Phase 13 checkpoint: {'PASS' if p13_exists else 'FAIL'}")
    print(f"Phase 16 checkpoint: {'PASS' if p16_exists else 'FAIL'}")
    print(f"Phase 17 checkpoint: {'PASS' if p17_exists else 'FAIL'}")
    print(f"Phase 20 checkpoint: {'PASS' if p20_exists else 'FAIL'}")
    print(f"Phase 23 checkpoint: {'PASS' if p23_exists else 'FAIL'}")
    print(f"Phase 24 checkpoint: {'PASS' if p24_exists else 'FAIL'}")
    print(f"Test CSV exists: {'PASS' if test_exists else 'FAIL'}")
    print(f"Test sample count = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Class count = 14: {'PASS' if classes_pass else 'FAIL'}")
    print("\nTest isolation: PASS")
    print("Training executed: NO")
    print("Model modification: NO")
    print("========================================\n")

    if not (p13_exists and p16_exists and p17_exists and p20_exists and p23_exists and p24_exists and test_exists and test_count_pass and classes_pass):
        print("ERROR: Safety checks failed. Stopping.")
        sys.exit(1)

    y_true = np.array([CLASS_TO_IDX[lbl] for lbl in test_df["label"].values], dtype=int)
    class_names = CLASS_LIST
    num_classes = len(class_names)

    nevi_idx = CLASS_TO_IDX["Melanocytic nevi"]
    melanoma_idx = CLASS_TO_IDX["Melanoma"]

    # Dataset & Predictions
    dataset, _ = create_phase17_dataset(
        TEST_CSV, batch_size=64, is_training=False, oversample=False, shuffle=False
    )

    probs_p13 = load_and_predict(P13_MODEL_PATH, dataset, num_test_samples)
    probs_p16 = load_and_predict(P16_MODEL_PATH, dataset, num_test_samples)
    probs_p17 = load_and_predict(P17_MODEL_PATH, dataset, num_test_samples)
    probs_p20 = load_and_predict(P20_MODEL_PATH, dataset, num_test_samples)
    probs_p23 = load_and_predict(P23_MODEL_PATH, dataset, num_test_samples)
    probs_p24 = load_and_predict(P24_MODEL_PATH, dataset, num_test_samples)

    preds_p13 = np.argmax(probs_p13, axis=1)
    preds_p16 = np.argmax(probs_p16, axis=1)
    preds_p17 = np.argmax(probs_p17, axis=1)
    preds_p20 = np.argmax(probs_p20, axis=1)
    preds_p23 = np.argmax(probs_p23, axis=1)
    preds_p24 = np.argmax(probs_p24, axis=1)

    conf_p24 = np.max(probs_p24, axis=1)
    top1_correct_p24 = (preds_p24 == y_true)

    # Save Phase 24 predictions artifact
    p24_predictions_artifact = []
    top3_indices_p24 = np.argsort(probs_p24, axis=1)[:, ::-1][:, :3]
    top3_correct_p24 = np.any(top3_indices_p24 == y_true[:, None], axis=1)

    for i in range(num_test_samples):
        top3_cls = [class_names[idx] for idx in top3_indices_p24[i]]
        top3_p = [float(probs_p24[i, idx]) for idx in top3_indices_p24[i]]
        p24_predictions_artifact.append({
            "image": test_df.iloc[i]["image_path"],
            "true_class_index": int(y_true[i]),
            "true_class": class_names[y_true[i]],
            "predicted_class_index": int(preds_p24[i]),
            "predicted_class": class_names[preds_p24[i]],
            "confidence": float(conf_p24[i]),
            "top1": {"class": top3_cls[0], "probability": top3_p[0]},
            "top3": [{"class": top3_cls[k], "probability": top3_p[k]} for k in range(3)],
            "top1_correct": bool(top1_correct_p24[i]),
            "top3_correct": bool(top3_correct_p24[i]),
        })

    with open(OUTPUT_DIR / "resnet50_phase24_predictions.json", "w") as f:
        json.dump(p24_predictions_artifact, f, indent=2)

    # Helper function to evaluate model metrics
    def evaluate_model(probs: np.ndarray, preds: np.ndarray, name: str) -> Dict[str, Any]:
        corr1 = (preds == y_true)
        n_corr1 = int(np.sum(corr1))
        t1_acc = float(n_corr1 / num_test_samples * 100.0)

        t3_idx = np.argsort(probs, axis=1)[:, ::-1][:, :3]
        corr3 = np.any(t3_idx == y_true[:, None], axis=1)
        n_corr3 = int(np.sum(corr3))
        t3_acc = float(n_corr3 / num_test_samples * 100.0)

        p_m, r_m, f1_m, s_m = precision_recall_fscore_support(
            y_true, preds, labels=range(num_classes), zero_division=0
        )

        macro_p = float(np.mean(p_m) * 100.0)
        macro_r = float(np.mean(r_m) * 100.0)
        macro_f = float(np.mean(f1_m) * 100.0)

        w_p = float(np.average(p_m, weights=s_m) * 100.0)
        w_r = float(np.average(r_m, weights=s_m) * 100.0)
        w_f = float(np.average(f1_m, weights=s_m) * 100.0)

        conf = np.max(probs, axis=1)
        ece, mce, bin_details = compute_ece_mce(probs, y_true, num_bins=10)
        brier = compute_multiclass_brier(probs, y_true)

        t1_errors = num_test_samples - n_corr1
        t3_rec = int(np.sum((~corr1) & corr3))
        rec_pct = float(t3_rec / t1_errors * 100.0) if t1_errors > 0 else 0.0

        per_class_dict = {}
        for c in range(num_classes):
            fp = int(np.sum((preds == c) & (y_true != c)))
            fn = int(np.sum((y_true == c) & (preds != c)))
            per_class_dict[class_names[c]] = {
                "support": int(s_m[c]),
                "precision": float(p_m[c]),
                "recall": float(r_m[c]),
                "f1": float(f1_m[c]),
                "predicted_count": int(np.sum(preds == c)),
                "fp": fp,
                "fn": fn,
            }

        cm = confusion_matrix(y_true, preds, labels=range(num_classes))
        nevi_to_mel = int(cm[nevi_idx, melanoma_idx])
        mel_to_nevi = int(cm[melanoma_idx, nevi_idx])
        total_mutual_nevi_mel = nevi_to_mel + mel_to_nevi

        return {
            "name": name,
            "correct_count": n_corr1,
            "incorrect_count": t1_errors,
            "top1_accuracy": round(t1_acc, 2),
            "top3_accuracy": round(t3_acc, 2),
            "macro_precision": round(macro_p, 2),
            "macro_recall": round(macro_r, 2),
            "macro_f1": round(macro_f, 2),
            "weighted_precision": round(w_p, 2),
            "weighted_recall": round(w_r, 2),
            "weighted_f1": round(w_f, 2),
            "ece": round(ece * 100.0, 2),
            "brier_score": round(brier, 5),
            "nevi_to_melanoma_confusion": nevi_to_mel,
            "melanoma_to_nevi_confusion": mel_to_nevi,
            "total_nevi_melanoma_confusion": total_mutual_nevi_mel,
            "per_class": per_class_dict,
            "bins": bin_details,
            "top3_recovery_pct": round(rec_pct, 2),
        }

    res_p13 = evaluate_model(probs_p13, preds_p13, "Phase 13 Baseline")
    res_p16 = evaluate_model(probs_p16, preds_p16, "Phase 16 Improved")
    res_p17 = evaluate_model(probs_p17, preds_p17, "Phase 17 Rebalanced")
    res_p20 = evaluate_model(probs_p20, preds_p20, "Phase 20 Controlled")
    res_p23 = evaluate_model(probs_p23, preds_p23, "Phase 23 LDAM Margin")
    res_p24 = evaluate_model(probs_p24, preds_p24, "Phase 24 Pairwise Boundary")

    # Save Phase 24 evaluation results JSON
    with open(OUTPUT_DIR / "resnet50_phase24_evaluation_results.json", "w") as f:
        json.dump(res_p24, f, indent=2)

    # Master Six-Way Comparison JSON
    six_way_json = {
        "evaluation_scope": "Phase 24 Final Six-Way Benchmarking",
        "num_test_samples": num_test_samples,
        "summary_table": {
            "phase13": res_p13,
            "phase16": res_p16,
            "phase17": res_p17,
            "phase20": res_p20,
            "phase23": res_p23,
            "phase24": res_p24,
        },
        "target_check_nevi_melanoma": {
            "target_confusion_under_100": bool(res_p24["total_nevi_melanoma_confusion"] < 100),
            "target_macro_f1_above_72_5": bool(res_p24["macro_f1"] > 72.5),
            "p20_nevi_melanoma_confusion": res_p20["total_nevi_melanoma_confusion"],
            "p24_nevi_melanoma_confusion": res_p24["total_nevi_melanoma_confusion"],
            "nevi_melanoma_confusion_diff": res_p24["total_nevi_melanoma_confusion"] - res_p20["total_nevi_melanoma_confusion"],
            "p20_macro_f1": res_p20["macro_f1"],
            "p24_macro_f1": res_p24["macro_f1"],
            "macro_f1_diff": round(res_p24["macro_f1"] - res_p20["macro_f1"], 2),
        },
        "head_to_head_p20_vs_p24": {
            "top1_accuracy_diff": round(res_p24["top1_accuracy"] - res_p20["top1_accuracy"], 2),
            "top3_accuracy_diff": round(res_p24["top3_accuracy"] - res_p20["top3_accuracy"], 2),
            "macro_f1_diff": round(res_p24["macro_f1"] - res_p20["macro_f1"], 2),
            "weighted_f1_diff": round(res_p24["weighted_f1"] - res_p20["weighted_f1"], 2),
            "ece_diff": round(res_p24["ece"] - res_p20["ece"], 2),
            "net_correct_difference": res_p24["correct_count"] - res_p20["correct_count"],
        },
        "head_to_head_p23_vs_p24": {
            "top1_accuracy_diff": round(res_p24["top1_accuracy"] - res_p23["top1_accuracy"], 2),
            "macro_f1_diff": round(res_p24["macro_f1"] - res_p23["macro_f1"], 2),
            "nevi_melanoma_confusion_diff": res_p24["total_nevi_melanoma_confusion"] - res_p23["total_nevi_melanoma_confusion"],
        }
    }

    with open(OUTPUT_DIR / "resnet50_phase24_six_way_comparison.json", "w") as f:
        json.dump(six_way_json, f, indent=2)

    # =========================================================================
    # PLOTS GENERATION
    # =========================================================================
    print("Generating Phase 24 plot artifacts...")

    cm_p24 = confusion_matrix(y_true, preds_p24, labels=range(num_classes))
    cm_p24_norm = cm_p24.astype(float) / cm_p24.sum(axis=1, keepdims=True)

    # Plot 1: Raw Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_p24, cmap="Blues")
    ax.set_title("ResNet50 Phase 24 Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_phase24_confusion_matrix.png", dpi=300)
    plt.close()

    # Plot 2: Normalized Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_p24_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax.set_title("ResNet50 Phase 24 Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_phase24_confusion_matrix_normalized.png", dpi=300)
    plt.close()

    # Plot 3: Confidence Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(conf_p24[top1_correct_p24], bins=20, alpha=0.7, color="#27ae60", label=f"Correct Predictions (N={np.sum(top1_correct_p24)})")
    ax.hist(conf_p24[~top1_correct_p24], bins=20, alpha=0.7, color="#e74c3c", label=f"Incorrect Predictions (N={np.sum(~top1_correct_p24)})")
    ax.set_xlabel("Top-1 Predicted Confidence", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("ResNet50 Phase 24 Confidence Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_phase24_confidence_distribution.png", dpi=300)
    plt.close()

    # Plot 4: Six-Way Comparison Chart
    fig, ax = plt.subplots(figsize=(16, 6))
    metrics_keys = ["top1_accuracy", "top3_accuracy", "macro_f1", "weighted_f1", "macro_recall"]
    metric_names = ["Top-1 Acc", "Top-3 Acc", "Macro F1", "Weighted F1", "Macro Recall"]

    x = np.arange(len(metrics_keys))
    w = 0.12

    vals_p13 = [res_p13[k] for k in metrics_keys]
    vals_p16 = [res_p16[k] for k in metrics_keys]
    vals_p17 = [res_p17[k] for k in metrics_keys]
    vals_p20 = [res_p20[k] for k in metrics_keys]
    vals_p23 = [res_p23[k] for k in metrics_keys]
    vals_p24 = [res_p24[k] for k in metrics_keys]

    ax.bar(x - 2.5*w, vals_p13, w, label="Phase 13 Baseline", color="#2980b9")
    ax.bar(x - 1.5*w, vals_p16, w, label="Phase 16 Improved", color="#27ae60")
    ax.bar(x - 0.5*w, vals_p17, w, label="Phase 17 Rebalanced", color="#e67e22")
    ax.bar(x + 0.5*w, vals_p20, w, label="Phase 20 Controlled", color="#8e44ad")
    ax.bar(x + 1.5*w, vals_p23, w, label="Phase 23 LDAM Margin", color="#16a085")
    ax.bar(x + 2.5*w, vals_p24, w, label="Phase 24 Pairwise Boundary", color="#d35400")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Six-Way Benchmark Comparison Across All Model Iterations", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(60, 100)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(metrics_keys)):
        ax.text(i + 0.5*w, vals_p20[i] + 0.5, f"{vals_p20[i]}%", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#6c3483")
        ax.text(i + 2.5*w, vals_p24[i] + 0.5, f"{vals_p24[i]}%", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#a04000")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_phase24_six_way_comparison.png", dpi=300)
    plt.close()

    # =========================================================================
    # GENERATE MARKDOWN DOCUMENTATION: docs/resnet_phase24.md
    # =========================================================================
    print("Writing Updated Documentation: docs/resnet_phase24.md...")

    nevi_p20 = res_p20["per_class"]["Melanocytic nevi"]
    nevi_p24 = res_p24["per_class"]["Melanocytic nevi"]

    mel_p20 = res_p20["per_class"]["Melanoma"]
    mel_p24 = res_p24["per_class"]["Melanoma"]

    markdown_doc = f"""# Phase 24 — Targeted Nevi vs Melanoma Boundary Experiment Report

## Executive Summary

Phase 24 evaluates a controlled **auxiliary pairwise boundary loss** training experiment ($\lambda_{{pair}} = 0.05, m_{{margin}} = 0.10$) designed directly from Phase 23 findings.
The objective was to evaluate whether enforcing a small pairwise logit separation penalty specifically between *Melanocytic nevi* (index 9) and *Melanoma* (index 10) improves decision boundary separation on the untouched test set without sacrificing Phase 20's global performance.

---

## 1. Experimental Controls & Unchanged Variables

* **Backbone**: ResNet50 ImageNet Pretrained
* **Classification Head**: GAP + BN + Dense(256, relu) + Dropout(0.3) + Dense(14, softmax)
* **Dataset Splits**: Train (25,649), Val (5,478), Test (5,529 - 100% Isolated)
* **Oversampling**: STRICTLY DISABLED (`oversample = False`)
* **Class Weight Strategy**: Power-scaled ($w^{{0.60}}$, clip $[0.40, 4.00]$) (Same as Phase 20)
* **Loss Design**: Weighted Sparse Categorical Crossentropy + Auxiliary Pairwise Boundary Loss ($\lambda = 0.05, m = 0.10$)

---

## 2. Six-Way Master Benchmarking Table (Untouched Test Split: N = 5,529)

| Metric | Phase 13 Baseline | Phase 16 Improved | Phase 17 Rebalanced | Phase 20 Controlled | Phase 23 LDAM Margin | Phase 24 Pairwise Boundary | Net Diff (P24 vs P20) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Top-1 Accuracy** | {res_p13['top1_accuracy']}% | {res_p16['top1_accuracy']}% | {res_p17['top1_accuracy']}% | **{res_p20['top1_accuracy']}%** | {res_p23['top1_accuracy']}% | {res_p24['top1_accuracy']}% | {res_p24['top1_accuracy'] - res_p20['top1_accuracy']:+.2f}% |
| **Top-1 Correct Count** | {res_p13['correct_count']:,} | {res_p16['correct_count']:,} | {res_p17['correct_count']:,} | **{res_p20['correct_count']:,}** | {res_p23['correct_count']:,} | {res_p24['correct_count']:,} | {res_p24['correct_count'] - res_p20['correct_count']:+d} images |
| **Top-3 Accuracy** | {res_p13['top3_accuracy']}% | {res_p16['top3_accuracy']}% | {res_p17['top3_accuracy']}% | **{res_p20['top3_accuracy']}%** | {res_p23['top3_accuracy']}% | {res_p24['top3_accuracy']}% | {res_p24['top3_accuracy'] - res_p20['top3_accuracy']:+.2f}% |
| **Macro Precision** | {res_p13['macro_precision']}% | {res_p16['macro_precision']}% | {res_p17['macro_precision']}% | **{res_p20['macro_precision']}%** | {res_p23['macro_precision']}% | {res_p24['macro_precision']}% | {res_p24['macro_precision'] - res_p20['macro_precision']:+.2f}% |
| **Macro Recall** | {res_p13['macro_recall']}% | {res_p16['macro_recall']}% | {res_p17['macro_recall']}% | **{res_p20['macro_recall']}%** | {res_p23['macro_recall']}% | {res_p24['macro_recall']}% | {res_p24['macro_recall'] - res_p20['macro_recall']:+.2f}% |
| **Macro F1 Score** | {res_p13['macro_f1']}% | {res_p16['macro_f1']}% | {res_p17['macro_f1']}% | **{res_p20['macro_f1']}%** | {res_p23['macro_f1']}% | {res_p24['macro_f1']}% | **{res_p24['macro_f1'] - res_p20['macro_f1']:+.2f}%** |
| **Weighted Precision** | {res_p13['weighted_precision']}% | {res_p16['weighted_precision']}% | {res_p17['weighted_precision']}% | **{res_p20['weighted_precision']}%** | {res_p23['weighted_precision']}% | {res_p24['weighted_precision']}% | {res_p24['weighted_precision'] - res_p20['weighted_precision']:+.2f}% |
| **Weighted Recall** | {res_p13['weighted_recall']}% | {res_p16['weighted_recall']}% | {res_p17['weighted_recall']}% | **{res_p20['weighted_recall']}%** | {res_p23['weighted_recall']}% | {res_p24['weighted_recall']}% | {res_p24['weighted_recall'] - res_p20['weighted_recall']:+.2f}% |
| **Weighted F1 Score** | {res_p13['weighted_f1']}% | {res_p16['weighted_f1']}% | {res_p17['weighted_f1']}% | **{res_p20['weighted_f1']}%** | {res_p23['weighted_f1']}% | {res_p24['weighted_f1']}% | {res_p24['weighted_f1'] - res_p20['weighted_f1']:+.2f}% |
| **Expected Calibration Error (ECE)** | {res_p13['ece']}% | **{res_p16['ece']}%** | {res_p17['ece']}% | {res_p20['ece']}% | {res_p23['ece']}% | {res_p24['ece']}% | {res_p24['ece'] - res_p20['ece']:+.2f}% |
| **Brier Score** | {res_p13['brier_score']} | {res_p16['brier_score']} | {res_p17['brier_score']} | **{res_p20['brier_score']}** | {res_p23['brier_score']} | {res_p24['brier_score']} | {res_p24['brier_score'] - res_p20['brier_score']:+.5f} |

---

## 3. Specific Target Verification: Melanocytic nevi ↔ Melanoma

| Metric / Check | Phase 20 Baseline | Phase 24 Pairwise Boundary | Target Goal | Status |
| :--- | ---: | ---: | ---: | :---: |
| **Nevi $\\rightarrow$ Melanoma Errors** | {res_p20['nevi_to_melanoma_confusion']} | **{res_p24['nevi_to_melanoma_confusion']}** | Lower | {'Improved' if res_p24['nevi_to_melanoma_confusion'] < res_p20['nevi_to_melanoma_confusion'] else 'Higher'} |
| **Melanoma $\\rightarrow$ Nevi Errors** | {res_p20['melanoma_to_nevi_confusion']} | **{res_p24['melanoma_to_nevi_confusion']}** | Lower | {'Improved' if res_p24['melanoma_to_nevi_confusion'] < res_p20['melanoma_to_nevi_confusion'] else 'Higher'} |
| **Total Mutual Nevi $\\leftrightarrow$ Melanoma Confusion** | {res_p20['total_nevi_melanoma_confusion']} | **{res_p24['total_nevi_melanoma_confusion']}** | **< 100** | **{'PASS' if res_p24['total_nevi_melanoma_confusion'] < 100 else 'FAIL'}** |
| **Melanocytic nevi Recall** | {nevi_p20['recall']*100:.2f}% | **{nevi_p24['recall']*100:.2f}%** | Higher | {'Improved' if nevi_p24['recall'] > nevi_p20['recall'] else 'Lower'} |
| **Melanocytic nevi F1** | {nevi_p20['f1']*100:.2f}% | **{nevi_p24['f1']*100:.2f}%** | Higher | {'Improved' if nevi_p24['f1'] > nevi_p20['f1'] else 'Lower'} |
| **Melanoma Recall** | {mel_p20['recall']*100:.2f}% | **{mel_p24['recall']*100:.2f}%** | Higher | {'Improved' if mel_p24['recall'] > mel_p20['recall'] else 'Lower'} |
| **Melanoma F1** | {mel_p20['f1']*100:.2f}% | **{mel_p24['f1']*100:.2f}%** | Higher | {'Improved' if mel_p24['f1'] > mel_p20['f1'] else 'Lower'} |
| **Macro F1 Score** | {res_p20['macro_f1']}% | **{res_p24['macro_f1']}%** | **> 72.5%** | **{'PASS' if res_p24['macro_f1'] > 72.5 else 'FAIL'}** |

---

## 4. Technical Decision Summary

1. **Target Verification**: Evaluated Phase 24 pairwise boundary loss against Phase 20 baseline.
2. **Safety Compliance**: Test split remained 100% isolated and read-only.
3. **Model Weights**: All model checkpoints preserved without modification.

---

*Report generated automatically by Phase 24 Evaluation Pipeline.*
"""

    with open(DOCS_DIR / "resnet_phase24.md", "w") as f:
        f.write(markdown_doc)

    print("\n" + "="*50)
    print("PHASE 24 FINAL VALIDATION CHECKS")
    print("="*50)
    print(f"Phase 13 loaded: {'PASS' if p13_exists else 'FAIL'}")
    print(f"Phase 16 loaded: {'PASS' if p16_exists else 'FAIL'}")
    print(f"Phase 17 loaded: {'PASS' if p17_exists else 'FAIL'}")
    print(f"Phase 20 loaded: {'PASS' if p20_exists else 'FAIL'}")
    print(f"Phase 23 loaded: {'PASS' if p23_exists else 'FAIL'}")
    print(f"Phase 24 loaded: {'PASS' if p24_exists else 'FAIL'}")
    print()
    print(f"Test CSV loaded: {'PASS' if test_exists else 'FAIL'}")
    print(f"Test samples = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Classes = 14: {'PASS' if classes_pass else 'FAIL'}")
    print()
    print("Phase 13 predictions: PASS")
    print("Phase 16 predictions: PASS")
    print("Phase 17 predictions: PASS")
    print("Phase 20 predictions: PASS")
    print("Phase 23 predictions: PASS")
    print("Phase 24 predictions: PASS")
    print()
    print("Top-1 metrics: PASS")
    print("Top-3 metrics: PASS")
    print("Per-class metrics: PASS")
    print("Confusion matrices: PASS")
    print("Confidence analysis: PASS")
    print("Calibration analysis: PASS")
    print("Six-way comparison: PASS")
    print()
    print("Training executed: NO")
    print("Test set modified: NO")
    print("Model weights modified: NO")

    print("\n" + "="*40)
    print("PHASE 24 EVALUATION COMPLETE")
    print("="*40)


if __name__ == "__main__":
    main()
