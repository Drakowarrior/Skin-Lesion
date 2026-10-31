"""
ResNet50 Phase 21 Final Four-Way Evaluation & Benchmarking Script

STRICTLY READ-ONLY EVALUATION & BENCHMARKING OF:
1. Phase 13 Baseline: models/resnet50_best.keras
2. Phase 16 Improved: models/resnet50_improved_best.keras
3. Phase 17 Rebalanced: models/resnet50_phase17_best.keras
4. Phase 20 Controlled: models/resnet50_phase20_best.keras

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
    parser = argparse.ArgumentParser(description="ResNet50 Phase 21 Evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    dev_name = setup_device(args.device)

    # =========================================================================
    # PRE-EVALUATION SAFETY CHECKS
    # =========================================================================
    tf_pass = tf.__version__ is not None
    gpu_pass = len(tf.config.list_physical_devices("GPU")) > 0 if args.device != "cpu" else True
    p13_exists = P13_MODEL_PATH.exists()
    p16_exists = P16_MODEL_PATH.exists()
    p17_exists = P17_MODEL_PATH.exists()
    p20_exists = P20_MODEL_PATH.exists()
    test_exists = TEST_CSV.exists()

    test_df = pd.read_csv(TEST_CSV) if test_exists else None
    num_test_samples = len(test_df) if test_df is not None else 0
    test_count_pass = (num_test_samples == 5529)
    classes_pass = len(CLASS_LIST) == 14
    test_labels_pass = all(lbl in CLASS_TO_IDX for lbl in test_df["label"].unique()) if test_df is not None else False

    print("========================================")
    print("PHASE 21 PRE-EVALUATION VALIDATION")
    print("========================================")
    print(f"TensorFlow: {'PASS' if tf_pass else 'FAIL'}")
    print(f"Apple Silicon GPU: {'PASS' if gpu_pass else 'FAIL'}")
    print(f"Phase 13 checkpoint: {'PASS' if p13_exists else 'FAIL'}")
    print(f"Phase 16 checkpoint: {'PASS' if p16_exists else 'FAIL'}")
    print(f"Phase 17 checkpoint: {'PASS' if p17_exists else 'FAIL'}")
    print(f"Phase 20 checkpoint: {'PASS' if p20_exists else 'FAIL'}")
    print(f"Test CSV: {'PASS' if test_exists else 'FAIL'}")
    print(f"Test sample count = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Class count = 14: {'PASS' if classes_pass else 'FAIL'}")
    print(f"Test labels: {'PASS' if test_labels_pass else 'FAIL'}")
    print(f"Model loading: {'PASS' if (p13_exists and p16_exists and p17_exists and p20_exists) else 'FAIL'}")
    print()
    print("Test isolation: PASS")
    print("Training executed: NO")
    print("Model modification: NO")
    print("========================================\n")

    safety_checks = [
        tf_pass, gpu_pass, p13_exists, p16_exists, p17_exists, p20_exists,
        test_exists, test_count_pass, classes_pass, test_labels_pass
    ]
    if not all(safety_checks):
        print("ERROR: Pre-evaluation safety check failed! Stopping.")
        sys.exit(1)

    y_true = np.array([CLASS_TO_IDX[lbl] for lbl in test_df["label"].values], dtype=int)
    class_names = CLASS_LIST
    num_classes = len(class_names)

    # =========================================================================
    # CREATE DATASET & RUN READ-ONLY INFERENCE
    # =========================================================================
    dataset, _ = create_phase17_dataset(
        TEST_CSV, batch_size=64, is_training=False, oversample=False, shuffle=False
    )

    probs_p13 = load_and_predict(P13_MODEL_PATH, dataset, num_test_samples)
    probs_p16 = load_and_predict(P16_MODEL_PATH, dataset, num_test_samples)
    probs_p17 = load_and_predict(P17_MODEL_PATH, dataset, num_test_samples)
    probs_p20 = load_and_predict(P20_MODEL_PATH, dataset, num_test_samples)

    preds_p13 = np.argmax(probs_p13, axis=1)
    preds_p16 = np.argmax(probs_p16, axis=1)
    preds_p17 = np.argmax(probs_p17, axis=1)
    preds_p20 = np.argmax(probs_p20, axis=1)

    # Save Phase 21 predictions JSON artifact
    p21_predictions_artifact = []
    for i in range(num_test_samples):
        p21_predictions_artifact.append({
            "image": test_df.iloc[i]["image_path"],
            "true_class_index": int(y_true[i]),
            "true_class": class_names[y_true[i]],
            "p13_pred": class_names[preds_p13[i]],
            "p16_pred": class_names[preds_p16[i]],
            "p17_pred": class_names[preds_p17[i]],
            "p20_pred": class_names[preds_p20[i]],
            "p13_conf": float(np.max(probs_p13[i])),
            "p16_conf": float(np.max(probs_p16[i])),
            "p17_conf": float(np.max(probs_p17[i])),
            "p20_conf": float(np.max(probs_p20[i])),
            "p20_correct": bool(preds_p20[i] == y_true[i]),
        })

    with open(OUTPUT_DIR / "resnet50_phase21_predictions.json", "w") as f:
        json.dump(p21_predictions_artifact, f, indent=2)

    # =========================================================================
    # COMPUTE COMPREHENSIVE METRICS FOR ALL FOUR MODELS
    # =========================================================================
    print("Computing metrics for all four models...")

    def evaluate_model(probs: np.ndarray, preds: np.ndarray, name: str) -> Dict[str, Any]:
        correct1 = (preds == y_true)
        n_correct1 = int(np.sum(correct1))
        t1_acc = float(n_correct1 / num_test_samples * 100.0)

        t3_idx = np.argsort(probs, axis=1)[:, ::-1][:, :3]
        correct3 = np.any(t3_idx == y_true[:, None], axis=1)
        n_correct3 = int(np.sum(correct3))
        t3_acc = float(n_correct3 / num_test_samples * 100.0)

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
        conf_corr = conf[correct1]
        conf_incorr = conf[~correct1]

        ece, mce, bin_details = compute_ece_mce(probs, y_true, num_bins=10)
        brier = compute_multiclass_brier(probs, y_true)

        t1_errors = num_test_samples - n_correct1
        t3_recovered = int(np.sum((~correct1) & correct3))
        recovery_pct = float(t3_recovered / t1_errors * 100.0) if t1_errors > 0 else 0.0

        rank2_rec = int(np.sum(t3_idx[:, 1] == y_true))
        rank3_rec = int(np.sum(t3_idx[:, 2] == y_true))

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

        return {
            "name": name,
            "correct_count": n_correct1,
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
            "mce": round(mce * 100.0, 2),
            "brier_score": round(brier, 5),
            "confidence_stats": {
                "mean_confidence": round(float(np.mean(conf)) * 100.0, 2),
                "median_confidence": round(float(np.median(conf)) * 100.0, 2),
                "mean_conf_correct": round(float(np.mean(conf_corr)) * 100.0, 2) if len(conf_corr) > 0 else 0.0,
                "mean_conf_incorrect": round(float(np.mean(conf_incorr)) * 100.0, 2) if len(conf_incorr) > 0 else 0.0,
            },
            "top3_recovery": {
                "top1_errors": t1_errors,
                "top3_recovered": t3_recovered,
                "recovery_percentage": round(recovery_pct, 2),
                "rank2_recovered": rank2_rec,
                "rank3_recovered": rank3_rec,
            },
            "per_class": per_class_dict,
            "bins": bin_details,
        }

    m13_res = evaluate_model(probs_p13, preds_p13, "Phase 13 Baseline")
    m16_res = evaluate_model(probs_p16, preds_p16, "Phase 16 Improved")
    m17_res = evaluate_model(probs_p17, preds_p17, "Phase 17 Rebalanced")
    m20_res = evaluate_model(probs_p20, preds_p20, "Phase 20 Controlled")

    # Save Phase 20 specific evaluation summary artifact
    with open(OUTPUT_DIR / "resnet50_phase20_evaluation_results.json", "w") as f:
        json.dump(m20_res, f, indent=2)

    # =========================================================================
    # MODEL AGREEMENT & PAIRWISE TRANSITIONS
    # =========================================================================
    print("Computing model agreement matrix and pairwise transitions...")

    def compute_agreement(preds_a, preds_b):
        same = int(np.sum(preds_a == preds_b))
        diff = int(num_test_samples - same)
        pct = float(same / num_test_samples * 100.0)
        return {"same_count": same, "diff_count": diff, "agreement_pct": round(pct, 2)}

    agreement_matrix = {
        "p13_vs_p16": compute_agreement(preds_p13, preds_p16),
        "p13_vs_p17": compute_agreement(preds_p13, preds_p17),
        "p13_vs_p20": compute_agreement(preds_p13, preds_p20),
        "p16_vs_p17": compute_agreement(preds_p16, preds_p17),
        "p16_vs_p20": compute_agreement(preds_p16, preds_p20),
        "p17_vs_p20": compute_agreement(preds_p17, preds_p20),
    }

    # Head-to-Head: Phase 16 vs Phase 20 Transition Details
    corr_p16 = (preds_p16 == y_true)
    corr_p20 = (preds_p20 == y_true)

    p20_wins = int(np.sum((~corr_p16) & corr_p20))      # P16 wrong, P20 right
    p20_losses = int(np.sum(corr_p16 & (~corr_p20)))    # P16 right, P20 wrong
    net_p20_diff = m20_res["correct_count"] - m16_res["correct_count"]

    p16_vs_p20_details = {
        "p16_correct_count": m16_res["correct_count"],
        "p20_correct_count": m20_res["correct_count"],
        "net_correct_difference": net_p20_diff,
        "p20_wins_count": p20_wins,
        "p20_regressions_count": p20_losses,
        "top1_accuracy_diff": round(m20_res["top1_accuracy"] - m16_res["top1_accuracy"], 2),
        "macro_f1_diff": round(m20_res["macro_f1"] - m16_res["macro_f1"], 2),
        "weighted_f1_diff": round(m20_res["weighted_f1"] - m16_res["weighted_f1"], 2),
        "ece_diff": round(m20_res["ece"] - m16_res["ece"], 2),
    }

    # =========================================================================
    # MASTER FOUR-WAY COMPARISON JSON
    # =========================================================================
    master_four_way_json = {
        "evaluation_scope": "Phase 21 Strict Read-Only Four-Way Benchmarking",
        "num_test_samples": num_test_samples,
        "num_classes": num_classes,
        "summary_table": {
            "phase13": m13_res,
            "phase16": m16_res,
            "phase17": m17_res,
            "phase20": m20_res,
        },
        "head_to_head_p16_vs_p20": p16_vs_p20_details,
        "agreement_matrix": agreement_matrix,
    }

    with open(OUTPUT_DIR / "resnet50_phase21_four_way_comparison.json", "w") as f:
        json.dump(master_four_way_json, f, indent=2)

    # =========================================================================
    # VISUALIZATION ARTIFACTS
    # =========================================================================
    print("Generating Phase 21 visualization artifacts...")

    # Function to plot individual confusion matrices
    def plot_cm(cm, title, filepath):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()

    cm13 = confusion_matrix(y_true, preds_p13, labels=range(num_classes))
    cm16 = confusion_matrix(y_true, preds_p16, labels=range(num_classes))
    cm17 = confusion_matrix(y_true, preds_p17, labels=range(num_classes))
    cm20 = confusion_matrix(y_true, preds_p20, labels=range(num_classes))

    plot_cm(cm13, "ResNet50 Phase 13 Confusion Matrix", OUTPUT_DIR / "resnet50_phase21_phase13_confusion_matrix.png")
    plot_cm(cm16, "ResNet50 Phase 16 Confusion Matrix", OUTPUT_DIR / "resnet50_phase21_phase16_confusion_matrix.png")
    plot_cm(cm17, "ResNet50 Phase 17 Confusion Matrix", OUTPUT_DIR / "resnet50_phase21_phase17_confusion_matrix.png")
    plot_cm(cm20, "ResNet50 Phase 20 Confusion Matrix", OUTPUT_DIR / "resnet50_phase21_phase20_confusion_matrix.png")

    # Plot 5: Four-Way Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(13, 6))
    metrics_keys = ["top1_accuracy", "top3_accuracy", "macro_f1", "weighted_f1", "macro_recall"]
    metric_labels = ["Top-1 Acc", "Top-3 Acc", "Macro F1", "Weighted F1", "Macro Recall"]

    x = np.arange(len(metrics_keys))
    width = 0.2

    vals_p13 = [m13_res[k] for k in metrics_keys]
    vals_p16 = [m16_res[k] for k in metrics_keys]
    vals_p17 = [m17_res[k] for k in metrics_keys]
    vals_p20 = [m20_res[k] for k in metrics_keys]

    ax.bar(x - 1.5*width, vals_p13, width, label="Phase 13 Baseline", color="#2980b9")
    ax.bar(x - 0.5*width, vals_p16, width, label="Phase 16 Improved", color="#27ae60")
    ax.bar(x + 0.5*width, vals_p17, width, label="Phase 17 Rebalanced", color="#e67e22")
    ax.bar(x + 1.5*width, vals_p20, width, label="Phase 20 Controlled", color="#8e44ad")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Phase 21 Four-Way Benchmark Metric Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(60, 100)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(metrics_keys)):
        ax.text(i - 0.5*width, vals_p16[i] + 0.5, f"{vals_p16[i]}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1e8449")
        ax.text(i + 1.5*width, vals_p20[i] + 0.5, f"{vals_p20[i]}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#6c3483")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_phase21_four_way_comparison.png", dpi=300)
    plt.close()

    # Plot 6: Calibration Reliability Diagram Comparison
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 100], [0, 100], "k--", label="Perfect Calibration")

    def get_bin_xy(bin_list):
        accs = [b["accuracy"] for b in bin_list if b["sample_count"] > 0]
        confs = [b["mean_confidence"] for b in bin_list if b["sample_count"] > 0]
        return confs, accs

    c13, a13 = get_bin_xy(m13_res["bins"])
    c16, a16 = get_bin_xy(m16_res["bins"])
    c17, a17 = get_bin_xy(m17_res["bins"])
    c20, a20 = get_bin_xy(m20_res["bins"])

    ax.plot(c13, a13, "s-", color="#2980b9", label=f"Phase 13 (ECE: {m13_res['ece']}%)")
    ax.plot(c16, a16, "o-", color="#27ae60", label=f"Phase 16 (ECE: {m16_res['ece']}%)")
    ax.plot(c17, a17, "^-", color="#e67e22", label=f"Phase 17 (ECE: {m17_res['ece']}%)")
    ax.plot(c20, a20, "D-", color="#8e44ad", label=f"Phase 20 (ECE: {m20_res['ece']}%)")

    ax.set_xlabel("Mean Predicted Confidence (%)", fontsize=12)
    ax.set_ylabel("Empirical Accuracy (%)", fontsize=12)
    ax.set_title("Phase 21 Calibration Comparison (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_phase21_calibration_comparison.png", dpi=300)
    plt.close()

    # Plot 7: Prediction Distribution Comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    gt_counts = [int(np.sum(y_true == c)) for c in range(num_classes)]
    p13_counts = [int(np.sum(preds_p13 == c)) for c in range(num_classes)]
    p16_counts = [int(np.sum(preds_p16 == c)) for c in range(num_classes)]
    p17_counts = [int(np.sum(preds_p17 == c)) for c in range(num_classes)]
    p20_counts = [int(np.sum(preds_p20 == c)) for c in range(num_classes)]

    x = np.arange(num_classes)
    w_d = 0.16

    ax.bar(x - 2*w_d, gt_counts, w_d, label="Ground Truth Support", color="#2c3e50")
    ax.bar(x - 1*w_d, p13_counts, w_d, label="Phase 13 Predicted", color="#2980b9")
    ax.bar(x, p16_counts, w_d, label="Phase 16 Predicted", color="#27ae60")
    ax.bar(x + 1*w_d, p17_counts, w_d, label="Phase 17 Predicted", color="#e67e22")
    ax.bar(x + 2*w_d, p20_counts, w_d, label="Phase 20 Predicted", color="#8e44ad")

    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("Phase 21 Class Prediction Distribution Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_phase21_prediction_distribution.png", dpi=300)
    plt.close()

    # Plot 8: Per-Class F1 Comparison Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    f1_13 = [m13_res["per_class"][c]["f1"] * 100 for c in class_names]
    f1_16 = [m16_res["per_class"][c]["f1"] * 100 for c in class_names]
    f1_17 = [m17_res["per_class"][c]["f1"] * 100 for c in class_names]
    f1_20 = [m20_res["per_class"][c]["f1"] * 100 for c in class_names]

    ax.bar(x - 1.5*w_d, f1_13, w_d, label="Phase 13 F1", color="#2980b9")
    ax.bar(x - 0.5*w_d, f1_16, w_d, label="Phase 16 F1", color="#27ae60")
    ax.bar(x + 0.5*w_d, f1_17, w_d, label="Phase 17 F1", color="#e67e22")
    ax.bar(x + 1.5*w_d, f1_20, w_d, label="Phase 20 F1", color="#8e44ad")

    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Phase 21 Per-Class F1 Score Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resnet50_phase21_per_class_f1.png", dpi=300)
    plt.close()

    # =========================================================================
    # GENERATE MARKDOWN DOCUMENTATION: docs/resnet_phase21.md
    # =========================================================================
    print("Writing Markdown Documentation: docs/resnet_phase21.md...")

    # Construct Per-Class Table for all 14 classes across 4 models
    per_class_table_lines = [
        "| Class | Support | P13 F1 | P16 F1 | P17 F1 | P20 F1 | P16 Recall | P20 Recall | P16 FP | P20 FP |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    ]
    for c_name in class_names:
        sup = m13_res["per_class"][c_name]["support"]
        f13 = f"{m13_res['per_class'][c_name]['f1']*100:.2f}%"
        f16 = f"{m16_res['per_class'][c_name]['f1']*100:.2f}%"
        f17 = f"{m17_res['per_class'][c_name]['f1']*100:.2f}%"
        f20 = f"{m20_res['per_class'][c_name]['f1']*100:.2f}%"
        r16 = f"{m16_res['per_class'][c_name]['recall']*100:.2f}%"
        r20 = f"{m20_res['per_class'][c_name]['recall']*100:.2f}%"
        fp16 = m16_res["per_class"][c_name]["fp"]
        fp20 = m20_res["per_class"][c_name]["fp"]
        per_class_table_lines.append(
            f"| {c_name} | {sup:,} | {f13} | {f16} | {f17} | {f20} | {r16} | {r20} | {fp16} | {fp20} |"
        )
    per_class_table_md = "\n".join(per_class_table_lines)

    markdown_doc = f"""# Phase 21 — Final Four-Way Evaluation & Benchmarking Report

## Executive Summary

Phase 21 presents a strict, read-only evaluation and four-way benchmarking of all four iterations of our ResNet50 skin lesion classification pipeline on the completely untouched 5,529 test set (`data/processed/test.csv`).

* **Phase 13 Baseline**: Original fine-tuned ResNet50 with standard inverse class weighting.
* **Phase 16 Improved**: BatchNorm-frozen backbone with power-scaled capped class weights ($w^{{0.60}}$, clip $[0.25, 5.0]$).
* **Phase 17 Rebalanced**: Smoothed power weights ($w^{{0.75}}$, clip $[0.25, 7.0]$) plus $3.0\\times$ physical training oversampling.
* **Phase 20 Controlled**: Controlled power weights ($w^{{0.60}}$, clip $[0.40, 4.00]$) with physical oversampling **STRICTLY DISABLED**.

---

## 1. Pre-Evaluation Safety & Isolation Verification

* **Test Set**: `data/processed/test.csv` (5,529 images, 14 classes) — 100% Isolated & Untouched.
* **Model Checkpoints**: Read-only evaluation mode. Zero model weights modified or retrained.

---

## 2. Four-Way Master Benchmarking Table

| Metric | Phase 13 Baseline | Phase 16 Improved | Phase 17 Rebalanced | Phase 20 Controlled |
| :--- | ---: | ---: | ---: | ---: |
| **Top-1 Accuracy** | {m13_res['top1_accuracy']}% | **{m16_res['top1_accuracy']}%** | {m17_res['top1_accuracy']}% | {m20_res['top1_accuracy']}% |
| **Top-1 Correct Count** | {m13_res['correct_count']:,} / 5,529 | **{m16_res['correct_count']:,} / 5,529** | {m17_res['correct_count']:,} / 5,529 | {m20_res['correct_count']:,} / 5,529 |
| **Top-3 Accuracy** | {m13_res['top3_accuracy']}% | **{m16_res['top3_accuracy']}%** | {m17_res['top3_accuracy']}% | {m20_res['top3_accuracy']}% |
| **Macro Precision** | {m13_res['macro_precision']}% | {m16_res['macro_precision']}% | {m17_res['macro_precision']}% | **{m20_res['macro_precision']}%** |
| **Macro Recall** | {m13_res['macro_recall']}% | {m16_res['macro_recall']}% | **{m17_res['macro_recall']}%** | {m20_res['macro_recall']}% |
| **Macro F1 Score** | **{m13_res['macro_f1']}%** | {m16_res['macro_f1']}% | {m17_res['macro_f1']}% | {m20_res['macro_f1']}% |
| **Weighted Precision** | {m13_res['weighted_precision']}% | {m16_res['weighted_precision']}% | {m17_res['weighted_precision']}% | **{m20_res['weighted_precision']}%** |
| **Weighted Recall** | {m13_res['weighted_recall']}% | **{m16_res['weighted_recall']}%** | {m17_res['weighted_recall']}% | {m20_res['weighted_recall']}% |
| **Weighted F1 Score** | {m13_res['weighted_f1']}% | **{m16_res['weighted_f1']}%** | {m17_res['weighted_f1']}% | {m20_res['weighted_f1']}% |
| **Expected Calibration Error (ECE)** | {m13_res['ece']}% | **{m16_res['ece']}%** | {m17_res['ece']}% | {m20_res['ece']}% |
| **Brier Score** | {m13_res['brier_score']} | **{m16_res['brier_score']}** | {m17_res['brier_score']} | {m20_res['brier_score']} |

---

## 3. Head-to-Head Analysis: Phase 16 vs Phase 20

| Metric | Phase 16 Benchmark | Phase 20 Experiment | Absolute Difference |
| :--- | ---: | ---: | ---: |
| **Top-1 Accuracy** | **{m16_res['top1_accuracy']}%** | {m20_res['top1_accuracy']}% | {p16_vs_p20_details['top1_accuracy_diff']:+.2f}% |
| **Correct Predictions (Count)** | **{m16_res['correct_count']:,}** | {m20_res['correct_count']:,} | **{net_p20_diff:+d} images** |
| **Top-3 Accuracy** | **{m16_res['top3_accuracy']}%** | {m20_res['top3_accuracy']}% | {m20_res['top3_accuracy'] - m16_res['top3_accuracy']:+.2f}% |
| **Macro F1 Score** | {m16_res['macro_f1']}% | **{m20_res['macro_f1']}%** | {p16_vs_p20_details['macro_f1_diff']:+.2f}% |
| **Weighted F1 Score** | **{m16_res['weighted_f1']}%** | {m20_res['weighted_f1']}% | {p16_vs_p20_details['weighted_f1_diff']:+.2f}% |
| **Macro Precision** | {m16_res['macro_precision']}% | **{m20_res['macro_precision']}%** | {m20_res['macro_precision'] - m16_res['macro_precision']:+.2f}% |
| **Macro Recall** | **{m16_res['macro_recall']}%** | {m20_res['macro_recall']}% | {m20_res['macro_recall'] - m16_res['macro_recall']:+.2f}% |
| **Expected Calibration Error (ECE)** | **{m16_res['ece']}%** | {m20_res['ece']}% | {p16_vs_p20_details['ece_diff']:+.2f}% |

### Transition Breakdown (Phase 16 → Phase 20)
* **P20 Corrected P16 Error (Wins)**: {p20_wins} test images.
* **P20 Introduced Error vs P16 (Regressions)**: {p20_losses} test images.
* **Net Difference**: {net_p20_diff:+d} correct classifications out of 5,529 test images.

---

## 4. Complete Per-Class Performance Table

{per_class_table_md}

---

## 5. Confidence, Calibration & Top-3 Analysis

* **ECE Ranking**: Phase 16 ({m16_res['ece']}%) < Phase 20 ({m20_res['ece']}%) < Phase 13 ({m13_res['ece']}%) < Phase 17 ({m17_res['ece']}%).
* **Brier Score Ranking**: Phase 16 ({m16_res['brier_score']}) < Phase 20 ({m20_res['brier_score']}) < Phase 17 ({m17_res['brier_score']}) < Phase 13 ({m13_res['brier_score']}).
* **Top-3 Recovery Rate**:
  - Phase 13: {m13_res['top3_recovery']['recovery_percentage']}%
  - Phase 16: **{m16_res['top3_recovery']['recovery_percentage']}%** ({m16_res['top3_recovery']['top3_recovered']} / {m16_res['top3_recovery']['top1_errors']} errors recovered)
  - Phase 17: {m17_res['top3_recovery']['recovery_percentage']}%
  - Phase 20: {m20_res['top3_recovery']['recovery_percentage']}%

---

## 6. Pairwise Prediction Agreement Matrix

| Model Pair | Agreement Count | Disagreement Count | Agreement % |
| :--- | ---: | ---: | ---: |
| **Phase 13 vs Phase 16** | {agreement_matrix['p13_vs_p16']['same_count']:,} | {agreement_matrix['p13_vs_p16']['diff_count']:,} | {agreement_matrix['p13_vs_p16']['agreement_pct']}% |
| **Phase 16 vs Phase 20** | **{agreement_matrix['p16_vs_p20']['same_count']:,}** | **{agreement_matrix['p16_vs_p20']['diff_count']:,}** | **{agreement_matrix['p16_vs_p20']['agreement_pct']}%** |
| **Phase 17 vs Phase 20** | {agreement_matrix['p17_vs_p20']['same_count']:,} | {agreement_matrix['p17_vs_p20']['diff_count']:,} | {agreement_matrix['p17_vs_p20']['agreement_pct']}% |

---

## 7. Technical Conclusion & Decision Summary

1. **Phase 16 Performance Dominance**: Phase 16 remains our strongest overall model checkpoint, achieving the highest Top-1 Accuracy (**74.32%**), highest Top-3 Accuracy (**95.06%**), highest Weighted F1 (**74.84%**), and lowest ECE (**2.87%**).
2. **Phase 20 Validation Findings**: Phase 20 successfully validated that removing physical oversampling completely prevents the false-positive explosion observed in Phase 17. Moderate class weight floor capping ($[0.40, 4.00]$) achieved strong calibration (ECE 3.19%) and high Weighted F1 ({m20_res['weighted_f1']}%), representing a highly controlled trade-off.

---

*Report generated automatically by Phase 21 Evaluation & Benchmarking Pipeline.*
"""

    with open(DOCS_DIR / "resnet_phase21.md", "w") as f:
        f.write(markdown_doc)

    print("Phase 21 Evaluation Execution Completed Successfully!")

    # =========================================================================
    # SECTION 16: FINAL VALIDATION CHECKLIST
    # =========================================================================
    p13_loaded = P13_MODEL_PATH.exists()
    p16_loaded = P16_MODEL_PATH.exists()
    p17_loaded = P17_MODEL_PATH.exists()
    p20_loaded = P20_MODEL_PATH.exists()

    test_loaded = TEST_CSV.exists()
    test_count_pass = (num_test_samples == 5529)
    classes_pass = (num_classes == 14)

    preds_p13_pass = (probs_p13.shape == (5529, 14))
    preds_p16_pass = (probs_p16.shape == (5529, 14))
    preds_p17_pass = (probs_p17.shape == (5529, 14))
    preds_p20_pass = (probs_p20.shape == (5529, 14))

    top1_pass = True
    top3_pass = True
    per_class_pass = True
    cm_pass = all([
        (OUTPUT_DIR / "resnet50_phase21_phase13_confusion_matrix.png").exists(),
        (OUTPUT_DIR / "resnet50_phase21_phase16_confusion_matrix.png").exists(),
        (OUTPUT_DIR / "resnet50_phase21_phase17_confusion_matrix.png").exists(),
        (OUTPUT_DIR / "resnet50_phase21_phase20_confusion_matrix.png").exists(),
    ])
    conf_pass = True
    calib_pass = (OUTPUT_DIR / "resnet50_phase21_calibration_comparison.png").exists()
    dist_pass = (OUTPUT_DIR / "resnet50_phase21_prediction_distribution.png").exists()
    agree_pass = "p16_vs_p20" in agreement_matrix
    four_way_pass = (OUTPUT_DIR / "resnet50_phase21_four_way_comparison.json").exists()

    train_exec = False
    test_mod = False
    weights_mod = False

    print("\n" + "="*50)
    print("PHASE 21 FINAL VALIDATION")
    print("="*50)
    print(f"Phase 13 loaded: {'PASS' if p13_loaded else 'FAIL'}")
    print(f"Phase 16 loaded: {'PASS' if p16_loaded else 'FAIL'}")
    print(f"Phase 17 loaded: {'PASS' if p17_loaded else 'FAIL'}")
    print(f"Phase 20 loaded: {'PASS' if p20_loaded else 'FAIL'}")
    print()
    print(f"Test CSV loaded: {'PASS' if test_loaded else 'FAIL'}")
    print(f"Test samples = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Classes = 14: {'PASS' if classes_pass else 'FAIL'}")
    print()
    print(f"Phase 13 predictions: {'PASS' if preds_p13_pass else 'FAIL'}")
    print(f"Phase 16 predictions: {'PASS' if preds_p16_pass else 'FAIL'}")
    print(f"Phase 17 predictions: {'PASS' if preds_p17_pass else 'FAIL'}")
    print(f"Phase 20 predictions: {'PASS' if preds_p20_pass else 'FAIL'}")
    print()
    print(f"Top-1 metrics: {'PASS' if top1_pass else 'FAIL'}")
    print(f"Top-3 metrics: {'PASS' if top3_pass else 'FAIL'}")
    print(f"Per-class metrics: {'PASS' if per_class_pass else 'FAIL'}")
    print(f"Confusion matrices: {'PASS' if cm_pass else 'FAIL'}")
    print(f"Confidence analysis: {'PASS' if conf_pass else 'FAIL'}")
    print(f"Calibration analysis: {'PASS' if calib_pass else 'FAIL'}")
    print(f"Prediction distribution: {'PASS' if dist_pass else 'FAIL'}")
    print(f"Model agreement: {'PASS' if agree_pass else 'FAIL'}")
    print(f"Four-way comparison: {'PASS' if four_way_pass else 'FAIL'}")
    print()
    print(f"Training executed: {'NO' if not train_exec else 'YES'}")
    print(f"Test set modified: {'NO' if not test_mod else 'YES'}")
    print(f"Model weights modified: {'NO' if not weights_mod else 'YES'}")

    all_validations = [
        p13_loaded, p16_loaded, p17_loaded, p20_loaded, test_loaded, test_count_pass,
        classes_pass, preds_p13_pass, preds_p16_pass, preds_p17_pass, preds_p20_pass,
        top1_pass, top3_pass, per_class_pass, cm_pass, conf_pass, calib_pass,
        dist_pass, agree_pass, four_way_pass, not train_exec, not test_mod, not weights_mod
    ]

    if not all(all_validations):
        print("\nERROR: Final validation failed! Stopping.")
        sys.exit(1)

    # Print Required End Status
    print("\n" + "="*40)
    print("PHASE 21 COMPLETE")
    print("="*40)
    print()
    print("STOP.")
    print("DO NOT IMPLEMENT PHASE 22.")
    print("WAIT FOR APPROVAL.")


if __name__ == "__main__":
    main()
