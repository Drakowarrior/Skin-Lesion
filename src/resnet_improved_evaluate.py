"""
ResNet50 Improved Model Test Evaluation & Baseline Comparison (Phase 16)

Evaluates the Phase 15 trained improved model (models/resnet50_improved_best.keras)
on the untouched test dataset (data/processed/test.csv, 5,529 images, 14 classes).

Generates:
1. Overall performance metrics (Top-1, Top-3, Macro/Weighted P, R, F1).
2. Confidence distribution & overconfidence analysis.
3. Top-3 recovery analysis (Rank #2 and Rank #3 counts).
4. Per-class metrics for all 14 classes.
5. Confusion matrices (raw counts & normalized percentages).
6. Detailed per-image prediction artifact (outputs/resnet50_improved_predictions.json).
7. Evaluation summary JSON (outputs/resnet50_improved_evaluation_results.json).
8. Objective baseline comparison JSON (outputs/resnet50_baseline_vs_improved.json).
"""

import argparse
import json
import sys
from pathlib import Path
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

from src.resnet_data import CLASS_LIST, CLASS_TO_IDX, IDX_TO_CLASS, create_dataset


IMPROVED_MODEL_PATH = Path("models/resnet50_improved_best.keras")
BASELINE_EVAL_JSON = Path("outputs/resnet50_evaluation_results.json")
TEST_CSV = Path("data/processed/test.csv")
OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet50 Improved Model Test Evaluation & Baseline Comparison"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(IMPROVED_MODEL_PATH),
        help="Path to trained improved model checkpoint (.keras)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(TEST_CSV),
        help="Path to test split CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Execution device ('auto', 'gpu', or 'cpu')",
    )
    return parser.parse_args()


def setup_device(device_choice: str) -> str:
    """Configure TensorFlow device context based on user argument."""
    if device_choice == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "CPU"
    
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"Apple Silicon GPU (Metal) - {gpus[0].name}"
    elif device_choice == "gpu":
        print("ERROR: GPU requested via --device gpu but no physical GPU detected.")
        sys.exit(1)
    else:
        return "CPU"


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: Path,
    title: str,
    fmt: str = "d",
    cmap=plt.cm.Blues,
):
    """Plot and save a clean, professional 14x14 confusion matrix graphic."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Class Label",
        xlabel="Predicted Class Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_str = f"{val:.1%}" if fmt == ".1%" else f"{val:d}"
            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_improved_resnet50(
    model_path: Path,
    test_csv: Path,
    batch_size: int = 32,
    device_choice: str = "auto",
):
    """Execute complete Improved ResNet50 evaluation and compare with baseline."""
    device_desc = setup_device(device_choice)

    if not model_path.exists():
        print(f"ERROR: Improved model checkpoint '{model_path}' not found.")
        sys.exit(1)

    if not test_csv.exists():
        print(f"ERROR: Test split CSV '{test_csv}' not found.")
        sys.exit(1)

    # Read Test CSV metadata
    test_df = pd.read_csv(test_csv)
    total_test_images = len(test_df)
    image_paths_list = test_df["image_path"].tolist()

    print("========================================")
    print("RESNET50 IMPROVED MODEL TEST EVALUATION")
    print("========================================")
    print(f"\nTensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {getattr(tf.keras, '__version__', '2.x/3.x')}")
    print(f"GPU Available: {'YES' if 'GPU' in device_desc else 'NO'}")
    print(f"Device: {device_desc}")
    print(f"\nImproved Model: {model_path}")
    print(f"Test Images: {total_test_images}")
    print(f"Classes: {len(CLASS_LIST)}")
    print("========================================\n")

    # Load Model & Dataset
    print("Loading Improved ResNet50 model checkpoint...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"Loading test dataset pipeline ({total_test_images} images)...")
    test_ds, _ = create_dataset(test_csv, batch_size=batch_size, is_training=False, shuffle=False)

    # Extract Ground Truth Labels
    y_true = test_df["label"].map(CLASS_TO_IDX).values

    # Perform Model Inference
    print("Executing model inference on test dataset...")
    y_probs = model.predict(test_ds, verbose=1)

    if y_probs.shape[0] != total_test_images or y_probs.shape[1] != len(CLASS_LIST):
        print(f"ERROR: Output prediction shape mismatch: expected ({total_test_images}, 14), got {y_probs.shape}")
        sys.exit(1)

    # Top-1 Predictions & Probabilities
    y_pred_top1 = np.argmax(y_probs, axis=1)
    top1_probs = np.max(y_probs, axis=1)

    # Top-3 Predictions & Rank Analysis
    top3_indices = np.argsort(y_probs, axis=1)[:, -3:][:, ::-1]  # Shape: (N, 3)
    
    top1_correct = (y_pred_top1 == y_true)
    top3_correct = np.array([y_true[i] in top3_indices[i] for i in range(total_test_images)])

    top1_incorrect_mask = ~top1_correct
    top1_incorrect_count = int(np.sum(top1_incorrect_mask))

    # Top-3 Recovery & Rank Breakdown
    top3_recovery_records = []
    rank2_count = 0
    rank3_count = 0

    for i in range(total_test_images):
        if not top1_correct[i] and top3_correct[i]:
            top3_recovery_records.append(i)
            if top3_indices[i, 1] == y_true[i]:
                rank2_count += 1
            elif top3_indices[i, 2] == y_true[i]:
                rank3_count += 1

    top3_recovery_count = len(top3_recovery_records)
    top3_recovery_pct_total = (top3_recovery_count / total_test_images * 100.0) if total_test_images > 0 else 0.0
    top3_recovery_pct_errors = (top3_recovery_count / top1_incorrect_count * 100.0) if top1_incorrect_count > 0 else 0.0

    # Overall Metric Calculations
    top1_acc = float(np.mean(top1_correct) * 100.0)
    top3_acc = float(np.mean(top3_correct) * 100.0)
    top3_improvement = top3_acc - top1_acc

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred_top1, average="macro")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred_top1, average="weighted")

    macro_p = float(macro_p * 100.0)
    macro_r = float(macro_r * 100.0)
    macro_f1 = float(macro_f1 * 100.0)
    weighted_p = float(weighted_p * 100.0)
    weighted_r = float(weighted_r * 100.0)
    weighted_f1 = float(weighted_f1 * 100.0)

    # Confidence Analysis
    avg_top1_prob = float(np.mean(top1_probs) * 100.0)
    median_top1_prob = float(np.median(top1_probs) * 100.0)
    min_top1_prob = float(np.min(top1_probs) * 100.0)
    max_top1_prob = float(np.max(top1_probs) * 100.0)

    correct_prob = float(np.mean(top1_probs[top1_correct]) * 100.0) if np.any(top1_correct) else 0.0
    incorrect_prob = float(np.mean(top1_probs[~top1_correct]) * 100.0) if np.any(~top1_correct) else 0.0

    below_50_cnt = int(np.sum(top1_probs < 0.50))
    below_70_cnt = int(np.sum(top1_probs < 0.70))
    below_90_cnt = int(np.sum(top1_probs < 0.90))

    below_50_pct = float(below_50_cnt / total_test_images * 100.0)
    below_70_pct = float(below_70_cnt / total_test_images * 100.0)
    below_90_pct = float(below_90_cnt / total_test_images * 100.0)

    # Overconfident Errors
    incorrect_top1_probs = top1_probs[~top1_correct] * 100.0
    overconf_50 = int(np.sum(incorrect_top1_probs >= 50.0))
    overconf_70 = int(np.sum(incorrect_top1_probs >= 70.0))
    overconf_80 = int(np.sum(incorrect_top1_probs >= 80.0))
    overconf_90 = int(np.sum(incorrect_top1_probs >= 90.0))
    overconf_95 = int(np.sum(incorrect_top1_probs >= 95.0))

    # Per-Class Metrics Calculation
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred_top1, labels=np.arange(len(CLASS_LIST))
    )

    per_class_metrics = {}
    print("\n----------------------------------------")
    print("PER-CLASS METRICS (IMPROVED RESNET50)")
    print("----------------------------------------")
    print(f"{'Class Name':<32} | {'Top-1 Acc':<9} | {'Top-3 Acc':<9} | {'Precision':<9} | {'Recall':<9} | {'F1':<7} | {'Support':<7}")
    print("-" * 100)

    for idx, cls_name in enumerate(CLASS_LIST):
        mask = (y_true == idx)
        cls_sup = int(per_class_support[idx])
        cls_top1_acc = float(np.mean(top1_correct[mask]) * 100.0) if cls_sup > 0 else 0.0
        cls_top3_acc = float(np.mean(top3_correct[mask]) * 100.0) if cls_sup > 0 else 0.0

        p_val = float(per_class_p[idx] * 100.0)
        r_val = float(per_class_r[idx] * 100.0)
        f1_val = float(per_class_f1[idx] * 100.0)

        per_class_metrics[cls_name] = {
            "top1_accuracy": round(cls_top1_acc, 2),
            "top3_accuracy": round(cls_top3_acc, 2),
            "precision": round(p_val, 2),
            "recall": round(r_val, 2),
            "f1_score": round(f1_val, 2),
            "support": cls_sup,
        }

        print(f"{cls_name:<32} | {cls_top1_acc:6.2f}%   | {cls_top3_acc:6.2f}%   | {p_val:6.2f}%   | {r_val:6.2f}%   | {f1_val:5.2f} | {cls_sup:<7}")

    print("-" * 100)

    # Save Output Visualizations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred_top1, labels=np.arange(len(CLASS_LIST)))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)

    cm_path = OUTPUT_DIR / "resnet50_improved_confusion_matrix.png"
    cm_norm_path = OUTPUT_DIR / "resnet50_improved_confusion_matrix_normalized.png"

    plot_confusion_matrix(cm, CLASS_LIST, cm_path, "Improved ResNet50 Confusion Matrix (Raw Counts)", fmt="d")
    plot_confusion_matrix(cm_norm, CLASS_LIST, cm_norm_path, "Improved ResNet50 Normalized Confusion Matrix (%)", fmt=".1%", cmap=plt.cm.Greens)

    print(f"\nSaved improved raw confusion matrix: {cm_path}")
    print(f"Saved improved normalized confusion matrix: {cm_norm_path}")

    # Generate Detailed Predictions JSON
    predictions_records = []
    for i in range(total_test_images):
        true_cls_name = IDX_TO_CLASS[y_true[i]]
        top1_cls_name = IDX_TO_CLASS[y_pred_top1[i]]
        top1_prob = float(top1_probs[i])

        top3_list = [
            {
                "class": IDX_TO_CLASS[top3_indices[i, k]],
                "probability": round(float(y_probs[i, top3_indices[i, k]]), 4),
            }
            for k in range(3)
        ]

        predictions_records.append({
            "image": image_paths_list[i],
            "true_class": true_cls_name,
            "top1": {
                "class": top1_cls_name,
                "probability": round(top1_prob, 4),
            },
            "top3": top3_list,
            "top1_correct": bool(top1_correct[i]),
            "top3_correct": bool(top3_correct[i]),
        })

    pred_json_path = OUTPUT_DIR / "resnet50_improved_predictions.json"
    with open(pred_json_path, "w") as f:
        json.dump(predictions_records, f, indent=2)
    print(f"Saved detailed improved predictions JSON: {pred_json_path}")

    # Generate Aggregated Evaluation Summary JSON
    summary_results = {
        "model": str(model_path),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "2.x/3.x"),
        "device": device_desc,
        "test_images": total_test_images,
        "num_classes": len(CLASS_LIST),
        "metrics": {
            "top1_accuracy": round(top1_acc, 2),
            "top3_accuracy": round(top3_acc, 2),
            "top3_improvement": round(top3_improvement, 2),
            "macro_precision": round(macro_p, 2),
            "macro_recall": round(macro_r, 2),
            "macro_f1": round(macro_f1, 2),
            "weighted_precision": round(weighted_p, 2),
            "weighted_recall": round(weighted_r, 2),
            "weighted_f1": round(weighted_f1, 2),
        },
        "confidence_analysis": {
            "average_top1_probability": round(avg_top1_prob, 2),
            "median_top1_probability": round(median_top1_prob, 2),
            "min_top1_probability": round(min_top1_prob, 2),
            "max_top1_probability": round(max_top1_prob, 2),
            "correct_prediction_probability": round(correct_prob, 2),
            "incorrect_prediction_probability": round(incorrect_prob, 2),
            "below_50_probability_count": below_50_cnt,
            "below_50_probability_percentage": round(below_50_pct, 2),
            "below_70_probability_count": below_70_cnt,
            "below_70_probability_percentage": round(below_70_pct, 2),
            "below_90_probability_count": below_90_cnt,
            "below_90_probability_percentage": round(below_90_pct, 2),
            "overconfident_errors": {
                "gte_50": overconf_50,
                "gte_70": overconf_70,
                "gte_80": overconf_80,
                "gte_90": overconf_90,
                "gte_95": overconf_95,
            },
        },
        "top3_analysis": {
            "top1_incorrect_top3_correct_count": top3_recovery_count,
            "top1_incorrect_top3_correct_percentage_of_total": round(top3_recovery_pct_total, 2),
            "top1_incorrect_top3_correct_percentage_of_errors": round(top3_recovery_pct_errors, 2),
            "rank_2_count": rank2_count,
            "rank_3_count": rank3_count,
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix_paths": {
            "raw": str(cm_path),
            "normalized": str(cm_norm_path),
        },
    }

    eval_json_path = OUTPUT_DIR / "resnet50_improved_evaluation_results.json"
    with open(eval_json_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Saved evaluation summary JSON: {eval_json_path}")

    # Objective Baseline Comparison (Phase 13 vs Phase 16)
    comparison_data = {}
    if BASELINE_EVAL_JSON.exists():
        with open(BASELINE_EVAL_JSON, "r") as f:
            baseline_results = json.load(f)

        base_metrics = baseline_results.get("metrics", {})
        base_per_class = baseline_results.get("per_class_metrics", {})
        base_conf = baseline_results.get("confidence_analysis", {})
        base_top3 = baseline_results.get("top3_analysis", {})

        top1_diff = round(top1_acc - base_metrics.get("top1_accuracy", 0.0), 2)
        top3_diff = round(top3_acc - base_metrics.get("top3_accuracy", 0.0), 2)
        macro_f1_diff = round(macro_f1 - base_metrics.get("macro_f1", 0.0), 2)
        weighted_f1_diff = round(weighted_f1 - base_metrics.get("weighted_f1", 0.0), 2)
        macro_p_diff = round(macro_p - base_metrics.get("macro_precision", 0.0), 2)
        macro_r_diff = round(macro_r - base_metrics.get("macro_recall", 0.0), 2)
        weighted_p_diff = round(weighted_p - base_metrics.get("weighted_precision", 0.0), 2)
        weighted_r_diff = round(weighted_r - base_metrics.get("weighted_recall", 0.0), 2)

        f1_improved = []
        f1_decreased = []
        recall_improved = []
        recall_decreased = []

        per_class_comparison = {}
        for cls in CLASS_LIST:
            imp_c = per_class_metrics.get(cls, {})
            base_c = base_per_class.get(cls, {})

            c_f1_diff = round(imp_c.get("f1_score", 0.0) - base_c.get("f1_score", 0.0), 2)
            c_rec_diff = round(imp_c.get("recall", 0.0) - base_c.get("recall", 0.0), 2)

            per_class_comparison[cls] = {
                "baseline_f1": base_c.get("f1_score", 0.0),
                "improved_f1": imp_c.get("f1_score", 0.0),
                "f1_diff": c_f1_diff,
                "baseline_recall": base_c.get("recall", 0.0),
                "improved_recall": imp_c.get("recall", 0.0),
                "recall_diff": c_rec_diff,
            }

            if c_f1_diff > 0:
                f1_improved.append(cls)
            elif c_f1_diff < 0:
                f1_decreased.append(cls)

            if c_rec_diff > 0:
                recall_improved.append(cls)
            elif c_rec_diff < 0:
                recall_decreased.append(cls)

        comparison_data = {
            "baseline_model": baseline_results.get("model", "models/resnet50_best.keras"),
            "improved_model": str(model_path),
            "test_images": total_test_images,
            "overall_differences": {
                "top1_accuracy": {"baseline": base_metrics.get("top1_accuracy"), "improved": top1_acc, "diff": top1_diff},
                "top3_accuracy": {"baseline": base_metrics.get("top3_accuracy"), "improved": top3_acc, "diff": top3_diff},
                "macro_f1": {"baseline": base_metrics.get("macro_f1"), "improved": macro_f1, "diff": macro_f1_diff},
                "weighted_f1": {"baseline": base_metrics.get("weighted_f1"), "improved": weighted_f1, "diff": weighted_f1_diff},
                "macro_precision": {"baseline": base_metrics.get("macro_precision"), "improved": macro_p, "diff": macro_p_diff},
                "macro_recall": {"baseline": base_metrics.get("macro_recall"), "improved": macro_r, "diff": macro_r_diff},
                "weighted_precision": {"baseline": base_metrics.get("weighted_precision"), "improved": weighted_p, "diff": weighted_p_diff},
                "weighted_recall": {"baseline": base_metrics.get("weighted_recall"), "improved": weighted_r, "diff": weighted_r_diff},
            },
            "confidence_changes": {
                "average_top1_probability": {
                    "baseline": base_conf.get("average_top1_probability"),
                    "improved": avg_top1_prob,
                    "diff": round(avg_top1_prob - base_conf.get("average_top1_probability", 0.0), 2),
                },
                "correct_prediction_probability": {
                    "baseline": base_conf.get("correct_prediction_probability"),
                    "improved": correct_prob,
                    "diff": round(correct_prob - base_conf.get("correct_prediction_probability", 0.0), 2),
                },
                "incorrect_prediction_probability": {
                    "baseline": base_conf.get("incorrect_prediction_probability"),
                    "improved": incorrect_prob,
                    "diff": round(incorrect_prob - base_conf.get("incorrect_prediction_probability", 0.0), 2),
                },
            },
            "top3_recovery_changes": {
                "baseline_count": base_top3.get("top1_incorrect_top3_correct_count"),
                "improved_count": top3_recovery_count,
                "diff": top3_recovery_count - base_top3.get("top1_incorrect_top3_correct_count", 0),
            },
            "per_class_summary": {
                "f1_improved_classes": f1_improved,
                "f1_decreased_classes": f1_decreased,
                "recall_improved_classes": recall_improved,
                "recall_decreased_classes": recall_decreased,
            },
            "per_class_comparison": per_class_comparison,
        }

        comp_json_path = OUTPUT_DIR / "resnet50_baseline_vs_improved.json"
        with open(comp_json_path, "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"Saved baseline vs improved comparison JSON: {comp_json_path}")

    # Print Terminal Summary Report
    print("\n========================================")
    print("IMPROVED RESNET50 TEST RESULTS")
    print("========================================")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"Top-3 Improvement: {top3_improvement:.2f} percentage points")
    print(f"\nMacro Precision: {macro_p:.2f}%")
    print(f"Macro Recall:    {macro_r:.2f}%")
    print(f"Macro F1 Score:  {macro_f1:.2f}%")
    print(f"\nWeighted Precision: {weighted_p:.2f}%")
    print(f"Weighted Recall:    {weighted_r:.2f}%")
    print(f"Weighted F1 Score:  {weighted_f1:.2f}%")

    print("\n========================================")
    print("CONFIDENCE ANALYSIS")
    print("========================================")
    print(f"Average Top-1 Probability: {avg_top1_prob:.2f}%")
    print(f"Median Top-1 Probability:  {median_top1_prob:.2f}%")
    print(f"Correct Prediction Probability: {correct_prob:.2f}%")
    print(f"Incorrect Prediction Probability: {incorrect_prob:.2f}%")
    print(f"\n<50% Probability: {below_50_cnt} / {total_test_images} ({below_50_pct:.2f}%)")
    print(f"<70% Probability: {below_70_cnt} / {total_test_images} ({below_70_pct:.2f}%)")
    print(f"<90% Probability: {below_90_cnt} / {total_test_images} ({below_90_pct:.2f}%)")

    print("\nOverconfident Errors:")
    print(f"  >= 50%: {overconf_50}")
    print(f"  >= 70%: {overconf_70}")
    print(f"  >= 80%: {overconf_80}")
    print(f"  >= 90%: {overconf_90}")
    print(f"  >= 95%: {overconf_95}")

    print("\n========================================")
    print("TOP-3 RECOVERY ANALYSIS")
    print("========================================")
    print(f"Top-1 Incorrect → Top-3 Correct:")
    print(f"  Total Count: {top3_recovery_count} / {total_test_images} ({top3_recovery_pct_total:.2f}% of test set)")
    print(f"  Percentage of Errors: {top3_recovery_pct_errors:.2f}%")
    print(f"  Rank #2 Recovery: {rank2_count}")
    print(f"  Rank #3 Recovery: {rank3_count}")

    if comparison_data:
        diffs = comparison_data["overall_differences"]
        print("\n========================================")
        print("BASELINE VS IMPROVED COMPARISON (TEST SET)")
        print("========================================")
        print(f"Top-1 Accuracy:  Baseline {diffs['top1_accuracy']['baseline']:.2f}% → Improved {top1_acc:.2f}% ({diffs['top1_accuracy']['diff']:+.2f}%)")
        print(f"Top-3 Accuracy:  Baseline {diffs['top3_accuracy']['baseline']:.2f}% → Improved {top3_acc:.2f}% ({diffs['top3_accuracy']['diff']:+.2f}%)")
        print(f"Macro F1 Score:  Baseline {diffs['macro_f1']['baseline']:.2f}% → Improved {macro_f1:.2f}% ({diffs['macro_f1']['diff']:+.2f}%)")
        print(f"Weighted F1:     Baseline {diffs['weighted_f1']['baseline']:.2f}% → Improved {weighted_f1:.2f}% ({diffs['weighted_f1']['diff']:+.2f}%)")

    print("\n========================================")
    print("STATUS: PASS")
    print("========================================")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        evaluate_improved_resnet50(
            model_path=Path(args.model_path),
            test_csv=Path(args.test_csv),
            batch_size=args.batch_size,
            device_choice=args.device,
        )
    )
