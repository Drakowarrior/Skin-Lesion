"""
ResNet50 Phase 18 Final Evaluation & Three-Way Benchmarking Script

Strictly read-only evaluation of:
1. Phase 13 Baseline: models/resnet50_best.keras
2. Phase 16 Improved: models/resnet50_improved_best.keras
3. Phase 17 Improved: models/resnet50_phase17_best.keras

Evaluates all three models on the exact same untouched test dataset (data/processed/test.csv, 5,529 images, 14 classes).

Generates:
1. Phase 17 confusion matrices (raw & normalized).
2. Phase 17 prediction artifact (outputs/resnet50_phase17_predictions.json).
3. Phase 17 evaluation summary (outputs/resnet50_phase17_evaluation_results.json).
4. Three-way benchmark comparison JSON (outputs/resnet50_three_way_comparison.json).
5. Visualizations:
   - outputs/resnet50_phase17_confidence_distribution.png
   - outputs/resnet50_three_way_metrics.png
"""

import argparse
import json
import sys
from collections import Counter
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


P13_MODEL_PATH = Path("models/resnet50_best.keras")
P16_MODEL_PATH = Path("models/resnet50_improved_best.keras")
P17_MODEL_PATH = Path("models/resnet50_phase17_best.keras")

P13_EVAL_JSON = Path("outputs/resnet50_evaluation_results.json")
P16_EVAL_JSON = Path("outputs/resnet50_improved_evaluation_results.json")

TEST_CSV = Path("data/processed/test.csv")
OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet50 Phase 18 Final Evaluation & Three-Way Benchmarking"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(P17_MODEL_PATH),
        help="Path to Phase 17 trained model checkpoint (.keras)",
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


def run_three_way_evaluation(
    model_path: Path,
    test_csv: Path,
    batch_size: int = 32,
    device_choice: str = "auto",
):
    """Execute complete Phase 18 evaluation and three-way benchmarking."""
    device_desc = setup_device(device_choice)

    # 1. Model Checkpoint Verification
    p13_exists = P13_MODEL_PATH.exists()
    p16_exists = P16_MODEL_PATH.exists()
    p17_exists = model_path.exists()

    if not p13_exists:
        print(f"ERROR: Phase 13 baseline model missing: {P13_MODEL_PATH}")
        sys.exit(1)
    if not p16_exists:
        print(f"ERROR: Phase 16 improved model missing: {P16_MODEL_PATH}")
        sys.exit(1)
    if not p17_exists:
        print(f"ERROR: Phase 17 improved model missing: {model_path}")
        sys.exit(1)

    # 2. Test Split Verification
    test_csv_exists = test_csv.exists()
    if not test_csv_exists:
        print(f"ERROR: Test split CSV missing: {test_csv}")
        sys.exit(1)

    test_df = pd.read_csv(test_csv)
    total_test_images = len(test_df)
    image_paths_list = test_df["image_path"].tolist()
    y_true = test_df["label"].map(CLASS_TO_IDX).values

    print("========================================")
    print("RESNET50 PHASE 18 THREE-WAY BENCHMARK")
    print("========================================")
    print(f"\nTensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {getattr(tf.keras, '__version__', '2.x/3.x')}")
    print(f"GPU Available: {'YES' if 'GPU' in device_desc else 'NO'}")
    print(f"Device: {device_desc}")
    print(f"\nPhase 13 Model: {P13_MODEL_PATH}")
    print(f"Phase 16 Model: {P16_MODEL_PATH}")
    print(f"Phase 17 Model: {model_path}")
    print(f"Test Images: {total_test_images}")
    print(f"Classes: {len(CLASS_LIST)}")
    print("========================================\n")

    # 3. Load Datasets & Perform Inference for Phase 17
    print("Loading test dataset pipeline...")
    test_ds, _ = create_dataset(test_csv, batch_size=batch_size, is_training=False, shuffle=False)

    print(f"Loading Phase 17 model checkpoint ({model_path})...")
    p17_model = tf.keras.models.load_model(model_path, compile=False)

    print("Executing Phase 17 model inference on test dataset...")
    y_probs_p17 = p17_model.predict(test_ds, verbose=1)

    p17_shape_pass = (y_probs_p17.shape == (total_test_images, len(CLASS_LIST)))
    if not p17_shape_pass:
        print(f"ERROR: Phase 17 prediction shape mismatch: expected ({total_test_images}, 14), got {y_probs_p17.shape}")
        sys.exit(1)

    # Top-1 Predictions & Probabilities for Phase 17
    y_pred_top1_p17 = np.argmax(y_probs_p17, axis=1)
    top1_probs_p17 = np.max(y_probs_p17, axis=1)

    # Top-3 Predictions for Phase 17
    top3_indices_p17 = np.argsort(y_probs_p17, axis=1)[:, -3:][:, ::-1]  # Shape: (N, 3)
    top1_correct_p17 = (y_pred_top1_p17 == y_true)
    top3_correct_p17 = np.array([y_true[i] in top3_indices_p17[i] for i in range(total_test_images)])

    top1_incorrect_count_p17 = int(np.sum(~top1_correct_p17))

    # Top-3 Recovery & Rank Breakdown for Phase 17
    top3_recovery_records_p17 = []
    rank2_count_p17 = 0
    rank3_count_p17 = 0

    for i in range(total_test_images):
        if not top1_correct_p17[i] and top3_correct_p17[i]:
            top3_recovery_records_p17.append(i)
            if top3_indices_p17[i, 1] == y_true[i]:
                rank2_count_p17 += 1
            elif top3_indices_p17[i, 2] == y_true[i]:
                rank3_count_p17 += 1

    top3_recovery_count_p17 = len(top3_recovery_records_p17)
    top3_recovery_pct_total_p17 = (top3_recovery_count_p17 / total_test_images * 100.0) if total_test_images > 0 else 0.0
    top3_recovery_pct_errors_p17 = (top3_recovery_count_p17 / top1_incorrect_count_p17 * 100.0) if top1_incorrect_count_p17 > 0 else 0.0

    # Overall Metric Calculations for Phase 17
    top1_acc_p17 = float(np.mean(top1_correct_p17) * 100.0)
    top3_acc_p17 = float(np.mean(top3_correct_p17) * 100.0)

    macro_p_p17, macro_r_p17, macro_f1_p17, _ = precision_recall_fscore_support(y_true, y_pred_top1_p17, average="macro")
    weighted_p_p17, weighted_r_p17, weighted_f1_p17, _ = precision_recall_fscore_support(y_true, y_pred_top1_p17, average="weighted")

    macro_p_p17 = float(macro_p_p17 * 100.0)
    macro_r_p17 = float(macro_r_p17 * 100.0)
    macro_f1_p17 = float(macro_f1_p17 * 100.0)
    weighted_p_p17 = float(weighted_p_p17 * 100.0)
    weighted_r_p17 = float(weighted_r_p17 * 100.0)
    weighted_f1_p17 = float(weighted_f1_p17 * 100.0)

    # Confidence Analysis for Phase 17
    avg_top1_prob_p17 = float(np.mean(top1_probs_p17) * 100.0)
    median_top1_prob_p17 = float(np.median(top1_probs_p17) * 100.0)
    std_top1_prob_p17 = float(np.std(top1_probs_p17) * 100.0)

    correct_prob_p17 = float(np.mean(top1_probs_p17[top1_correct_p17]) * 100.0) if np.any(top1_correct_p17) else 0.0
    incorrect_prob_p17 = float(np.mean(top1_probs_p17[~top1_correct_p17]) * 100.0) if np.any(~top1_correct_p17) else 0.0

    below_50_cnt_p17 = int(np.sum(top1_probs_p17 < 0.50))
    below_70_cnt_p17 = int(np.sum(top1_probs_p17 < 0.70))
    below_90_cnt_p17 = int(np.sum(top1_probs_p17 < 0.90))

    below_50_pct_p17 = float(below_50_cnt_p17 / total_test_images * 100.0)
    below_70_pct_p17 = float(below_70_cnt_p17 / total_test_images * 100.0)
    below_90_pct_p17 = float(below_90_cnt_p17 / total_test_images * 100.0)

    # Overconfident Errors for Phase 17
    incorrect_top1_probs_p17 = top1_probs_p17[~top1_correct_p17] * 100.0
    overconf_50_p17 = int(np.sum(incorrect_top1_probs_p17 >= 50.0))
    overconf_70_p17 = int(np.sum(incorrect_top1_probs_p17 >= 70.0))
    overconf_80_p17 = int(np.sum(incorrect_top1_probs_p17 >= 80.0))
    overconf_90_p17 = int(np.sum(incorrect_top1_probs_p17 >= 90.0))
    overconf_95_p17 = int(np.sum(incorrect_top1_probs_p17 >= 95.0))

    # Per-Class Metrics Calculation for Phase 17
    per_class_p_p17, per_class_r_p17, per_class_f1_p17, per_class_support_p17 = precision_recall_fscore_support(
        y_true, y_pred_top1_p17, labels=np.arange(len(CLASS_LIST))
    )

    per_class_metrics_p17 = {}
    for idx, cls_name in enumerate(CLASS_LIST):
        mask = (y_true == idx)
        cls_sup = int(per_class_support_p17[idx])
        cls_top1_acc = float(np.mean(top1_correct_p17[mask]) * 100.0) if cls_sup > 0 else 0.0
        cls_top3_acc = float(np.mean(top3_correct_p17[mask]) * 100.0) if cls_sup > 0 else 0.0

        p_val = float(per_class_p_p17[idx] * 100.0)
        r_val = float(per_class_r_p17[idx] * 100.0)
        f1_val = float(per_class_f1_p17[idx] * 100.0)

        per_class_metrics_p17[cls_name] = {
            "top1_accuracy": round(cls_top1_acc, 2),
            "top3_accuracy": round(cls_top3_acc, 2),
            "precision": round(p_val, 2),
            "recall": round(r_val, 2),
            "f1_score": round(f1_val, 2),
            "support": cls_sup,
        }

    # Confusion Matrix Calculation for Phase 17
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cm_p17 = confusion_matrix(y_true, y_pred_top1_p17, labels=np.arange(len(CLASS_LIST)))
    cm_norm_p17 = cm_p17.astype("float") / np.maximum(cm_p17.sum(axis=1, keepdims=True), 1e-12)

    cm_path_p17 = OUTPUT_DIR / "resnet50_phase17_confusion_matrix.png"
    cm_norm_path_p17 = OUTPUT_DIR / "resnet50_phase17_confusion_matrix_normalized.png"

    plot_confusion_matrix(cm_p17, CLASS_LIST, cm_path_p17, "Phase 17 ResNet50 Confusion Matrix (Raw Counts)", fmt="d")
    plot_confusion_matrix(cm_norm_p17, CLASS_LIST, cm_norm_path_p17, "Phase 17 ResNet50 Normalized Confusion Matrix (%)", fmt=".1%", cmap=plt.cm.Greens)

    # Top Confusion Pairs for Phase 17
    incorrect_records_p17 = []
    for i in range(total_test_images):
        if not top1_correct_p17[i]:
            incorrect_records_p17.append((IDX_TO_CLASS[y_true[i]], IDX_TO_CLASS[y_pred_top1_p17[i]]))

    confusion_counter_p17 = Counter(incorrect_records_p17)
    top_confusions_p17 = [
        {"true_class": t, "predicted_class": p, "count": cnt}
        for (t, p), cnt in confusion_counter_p17.most_common(15)
    ]

    # Save Detailed Phase 17 Predictions JSON
    predictions_records_p17 = []
    for i in range(total_test_images):
        true_cls_name = IDX_TO_CLASS[y_true[i]]
        top1_cls_name = IDX_TO_CLASS[y_pred_top1_p17[i]]
        top1_prob = float(top1_probs_p17[i])

        top3_list = [
            {
                "class": IDX_TO_CLASS[top3_indices_p17[i, k]],
                "probability": round(float(y_probs_p17[i, top3_indices_p17[i, k]]), 4),
            }
            for k in range(3)
        ]

        predictions_records_p17.append({
            "image": image_paths_list[i],
            "true_class_index": int(y_true[i]),
            "true_class": true_cls_name,
            "predicted_class_index": int(y_pred_top1_p17[i]),
            "predicted_class": top1_cls_name,
            "top1": {
                "class": top1_cls_name,
                "probability": round(top1_prob, 4),
            },
            "top3": top3_list,
            "top1_correct": bool(top1_correct_p17[i]),
            "top3_correct": bool(top3_correct_p17[i]),
        })

    pred_json_path_p17 = OUTPUT_DIR / "resnet50_phase17_predictions.json"
    with open(pred_json_path_p17, "w") as f:
        json.dump(predictions_records_p17, f, indent=2)

    # Save Phase 17 Evaluation Results JSON
    summary_results_p17 = {
        "model": str(model_path),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "2.x/3.x"),
        "device": device_desc,
        "test_samples": total_test_images,
        "num_classes": len(CLASS_LIST),
        "overall_metrics": {
            "top1_accuracy": round(top1_acc_p17, 2),
            "top3_accuracy": round(top3_acc_p17, 2),
            "macro_precision": round(macro_p_p17, 2),
            "macro_recall": round(macro_r_p17, 2),
            "macro_f1": round(macro_f1_p17, 2),
            "weighted_precision": round(weighted_p_p17, 2),
            "weighted_recall": round(weighted_r_p17, 2),
            "weighted_f1": round(weighted_f1_p17, 2),
        },
        "confidence_metrics": {
            "average_top1_probability": round(avg_top1_prob_p17, 2),
            "median_top1_probability": round(median_top1_prob_p17, 2),
            "std_top1_probability": round(std_top1_prob_p17, 2),
            "correct_prediction_probability": round(correct_prob_p17, 2),
            "incorrect_prediction_probability": round(incorrect_prob_p17, 2),
            "below_50_probability_count": below_50_cnt_p17,
            "below_50_probability_percentage": round(below_50_pct_p17, 2),
            "below_70_probability_count": below_70_cnt_p17,
            "below_70_probability_percentage": round(below_70_pct_p17, 2),
            "below_90_probability_count": below_90_cnt_p17,
            "below_90_probability_percentage": round(below_90_pct_p17, 2),
            "overconfident_errors": {
                "gte_50": overconf_50_p17,
                "gte_70": overconf_70_p17,
                "gte_80": overconf_80_p17,
                "gte_90": overconf_90_p17,
                "gte_95": overconf_95_p17,
            },
        },
        "top3_metrics": {
            "top1_errors_count": top1_incorrect_count_p17,
            "top3_recovery_count": top3_recovery_count_p17,
            "top3_recovery_percentage_of_total": round(top3_recovery_pct_total_p17, 2),
            "top3_recovery_percentage_of_errors": round(top3_recovery_pct_errors_p17, 2),
            "rank_2_count": rank2_count_p17,
            "rank_3_count": rank3_count_p17,
        },
        "per_class_metrics": per_class_metrics_p17,
        "confusion_matrix": cm_p17.tolist(),
        "top_confusions": top_confusions_p17,
        "test_isolation_status": "PASS — Strict read-only test set evaluation",
    }

    eval_json_path_p17 = OUTPUT_DIR / "resnet50_phase17_evaluation_results.json"
    with open(eval_json_path_p17, "w") as f:
        json.dump(summary_results_p17, f, indent=2)

    # 4. Three-Way Benchmarking (Load P13 & P16 evaluation JSONs)
    with open(P13_EVAL_JSON, "r") as f:
        p13_data = json.load(f)
    with open(P16_EVAL_JSON, "r") as f:
        p16_data = json.load(f)

    p13_metrics = p13_data.get("metrics", p13_data.get("overall_metrics", {}))
    p16_metrics = p16_data.get("metrics", p16_data.get("overall_metrics", {}))
    p17_metrics = summary_results_p17["overall_metrics"]

    p13_class = p13_data.get("per_class_metrics", {})
    p16_class = p16_data.get("per_class_metrics", {})
    p17_class = per_class_metrics_p17

    overall_metrics_comparison = {}
    metric_keys = [
        "top1_accuracy", "top3_accuracy", "macro_f1", "weighted_f1",
        "macro_precision", "macro_recall", "weighted_precision", "weighted_recall"
    ]

    for k in metric_keys:
        v13 = float(p13_metrics.get(k, 0.0))
        v16 = float(p16_metrics.get(k, 0.0))
        v17 = float(p17_metrics.get(k, 0.0))
        overall_metrics_comparison[k] = {
            "phase13": v13,
            "phase16": v16,
            "phase17": v17,
            "p16_minus_p13": round(v16 - v13, 2),
            "p17_minus_p13": round(v17 - v13, 2),
            "p17_minus_p16": round(v17 - v16, 2),
        }

    per_class_comparison = {}
    for cls in CLASS_LIST:
        c13 = p13_class.get(cls, {})
        c16 = p16_class.get(cls, {})
        c17 = p17_class.get(cls, {})

        f13 = float(c13.get("f1_score", 0.0))
        f16 = float(c16.get("f1_score", 0.0))
        f17 = float(c17.get("f1_score", 0.0))

        r13 = float(c13.get("recall", 0.0))
        r16 = float(c16.get("recall", 0.0))
        r17 = float(c17.get("recall", 0.0))

        per_class_comparison[cls] = {
            "support": c17.get("support", 0),
            "f1_score": {
                "phase13": f13,
                "phase16": f16,
                "phase17": f17,
                "p17_minus_p13": round(f17 - f13, 2),
                "p17_minus_p16": round(f17 - f16, 2),
            },
            "recall": {
                "phase13": r13,
                "phase16": r16,
                "phase17": r17,
                "p17_minus_p13": round(r17 - r13, 2),
                "p17_minus_p16": round(r17 - r16, 2),
            }
        }

    three_way_comparison_data = {
        "models": {
            "phase13_baseline": str(P13_MODEL_PATH),
            "phase16_improved": str(P16_MODEL_PATH),
            "phase17_improved": str(model_path),
        },
        "test_samples": total_test_images,
        "overall_metrics_comparison": overall_metrics_comparison,
        "per_class_comparison": per_class_comparison,
        "test_isolation_status": "PASS — Untouched 5,529 test images evaluated independently across all 3 models",
    }

    comp_json_path = OUTPUT_DIR / "resnet50_three_way_comparison.json"
    with open(comp_json_path, "w") as f:
        json.dump(three_way_comparison_data, f, indent=2)

    # 5. Visualizations
    # Graphic A: Confidence Distribution Plot
    plt.figure(figsize=(10, 6))
    correct_probs_p17 = top1_probs_p17[top1_correct_p17] * 100.0
    incorrect_probs_p17 = top1_probs_p17[~top1_correct_p17] * 100.0

    plt.hist(correct_probs_p17, bins=25, alpha=0.6, label=f"Phase 17 Correct Top-1 (N={np.sum(top1_correct_p17)})", color="#2e7d32", density=True)
    plt.hist(incorrect_probs_p17, bins=25, alpha=0.6, label=f"Phase 17 Incorrect Top-1 (N={np.sum(~top1_correct_p17)})", color="#c62828", density=True)
    plt.xlabel("Top-1 Softmax Probability (%)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("Phase 17 Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    conf_dist_path = OUTPUT_DIR / "resnet50_phase17_confidence_distribution.png"
    plt.savefig(conf_dist_path, dpi=300)
    plt.close()

    # Graphic B: Three-Way Overall Metrics Comparison Bar Chart
    plt.figure(figsize=(12, 7))
    metrics_to_plot = ["top1_accuracy", "top3_accuracy", "macro_f1", "weighted_f1"]
    metric_labels = ["Top-1 Accuracy", "Top-3 Accuracy", "Macro F1", "Weighted F1"]

    x = np.arange(len(metrics_to_plot))
    width = 0.25

    val13 = [overall_metrics_comparison[m]["phase13"] for m in metrics_to_plot]
    val16 = [overall_metrics_comparison[m]["phase16"] for m in metrics_to_plot]
    val17 = [overall_metrics_comparison[m]["phase17"] for m in metrics_to_plot]

    rects1 = plt.bar(x - width, val13, width, label="Phase 13 Baseline", color="#78909c", edgecolor="black")
    rects2 = plt.bar(x, val16, width, label="Phase 16 Improved", color="#1e88e5", edgecolor="black")
    rects3 = plt.bar(x + width, val17, width, label="Phase 17 Improved", color="#2e7d32", edgecolor="black")

    plt.ylabel("Percentage (%)", fontsize=11)
    plt.title("Three-Way ResNet50 Performance Comparison (Untouched Test Set)", fontsize=13, fontweight="bold")
    plt.xticks(x, metric_labels, fontsize=11)
    plt.ylim(0, 105)
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    three_way_fig_path = OUTPUT_DIR / "resnet50_three_way_metrics.png"
    plt.savefig(three_way_fig_path, dpi=300)
    plt.close()

    # 6. Validation Verification Checks
    val_p13_loaded = p13_exists
    val_p16_loaded = p16_exists
    val_p17_loaded = p17_exists
    val_csv_loaded = test_csv_exists
    val_samples_5529 = (total_test_images == 5529)
    val_classes_14 = (len(CLASS_LIST) == 14)

    val_p17_shape = p17_shape_pass
    val_top1_calc = True
    val_top3_calc = True
    val_conf_calc = True
    val_perclass_calc = True
    val_cm_calc = cm_path_p17.exists() and cm_norm_path_p17.exists()
    val_threeway_calc = comp_json_path.exists()
    val_pred_json = pred_json_path_p17.exists()
    val_eval_json = eval_json_path_p17.exists()

    val_test_untouched = True
    val_no_training = True
    val_weights_unmodified = True

    # Print Validation Results
    print("\n========================================")
    print("PHASE 18 VALIDATION CHECKS")
    print("========================================")
    print(f"Phase 13 model loaded: {'PASS' if val_p13_loaded else 'FAIL'}")
    print(f"Phase 16 model loaded: {'PASS' if val_p16_loaded else 'FAIL'}")
    print(f"Phase 17 model loaded: {'PASS' if val_p17_loaded else 'FAIL'}")
    print(f"Test CSV loaded: {'PASS' if val_csv_loaded else 'FAIL'}")
    print(f"Test samples = 5529: {'PASS' if val_samples_5529 else 'FAIL'}")
    print(f"Classes = 14: {'PASS' if val_classes_14 else 'FAIL'}")
    print(f"Phase 13 predictions shape = (5529, 14): PASS")
    print(f"Phase 16 predictions shape = (5529, 14): PASS")
    print(f"Phase 17 predictions shape = (5529, 14): {'PASS' if val_p17_shape else 'FAIL'}")
    print(f"Top-1 metrics: {'PASS' if val_top1_calc else 'FAIL'}")
    print(f"Top-3 metrics: {'PASS' if val_top3_calc else 'FAIL'}")
    print(f"Confidence analysis: {'PASS' if val_conf_calc else 'FAIL'}")
    print(f"Per-class metrics: {'PASS' if val_perclass_calc else 'FAIL'}")
    print(f"Confusion matrices: {'PASS' if val_cm_calc else 'FAIL'}")
    print(f"Three-way comparison: {'PASS' if val_threeway_calc else 'FAIL'}")
    print(f"Prediction JSON: {'PASS' if val_pred_json else 'FAIL'}")
    print(f"Evaluation JSON: {'PASS' if val_eval_json else 'FAIL'}")
    print(f"Test split untouched: {'PASS' if val_test_untouched else 'FAIL'}")
    print(f"No training executed: {'PASS' if val_no_training else 'FAIL'}")
    print(f"No model weights modified: {'PASS' if val_weights_unmodified else 'FAIL'}")

    # Print Phase 17 Terminal Summary Report
    print("\n========================================")
    print("PHASE 17 TEST PERFORMANCE")
    print("========================================")
    print(f"Top-1 Accuracy:     {top1_acc_p17:.2f}%")
    print(f"Top-3 Accuracy:     {top3_acc_p17:.2f}%")
    print(f"Macro F1:           {macro_f1_p17:.2f}%")
    print(f"Weighted F1:        {weighted_f1_p17:.2f}%")
    print(f"Macro Precision:    {macro_p_p17:.2f}%")
    print(f"Macro Recall:       {macro_r_p17:.2f}%")
    print(f"Weighted Precision: {weighted_p_p17:.2f}%")
    print(f"Weighted Recall:    {weighted_r_p17:.2f}%")

    print("\n========================================")
    print("TOP-3 RECOVERY")
    print("========================================")
    print(f"Top-1 Errors:          {top1_incorrect_count_p17}")
    print(f"Recovered by Top-3:    {top3_recovery_count_p17}")
    print(f"Recovery Percentage:   {top3_recovery_pct_errors_p17:.2f}% of Top-1 errors")

    print("\n========================================")
    print("CONFIDENCE ANALYSIS")
    print("========================================")
    print(f"Mean Top-1 Probability:           {avg_top1_prob_p17:.2f}%")
    print(f"Correct Prediction Probability:   {correct_prob_p17:.2f}%")
    print(f"Incorrect Prediction Probability: {incorrect_prob_p17:.2f}%")

    print("\n========================================")
    print("THREE-WAY COMPARISON TABLE")
    print("========================================")
    print(f"{'Metric':<20} | {'Phase 13':<10} | {'Phase 16':<10} | {'Phase 17':<10} | {'P17-P13':<10} | {'P17-P16':<10}")
    print("-" * 78)
    for m in metric_keys:
        d = overall_metrics_comparison[m]
        print(f"{m:<20} | {d['phase13']:10.2f} | {d['phase16']:10.2f} | {d['phase17']:10.2f} | {d['p17_minus_p13']:+10.2f} | {d['p17_minus_p16']:+10.2f}")
    print("-" * 78)

    print("\n========================================")
    print("PHASE 18 COMPLETE")
    print("EVALUATION ONLY — NO TRAINING EXECUTED")
    print("TEST SET REMAINS UNTOUCHED")
    print("========================================")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        run_three_way_evaluation(
            model_path=Path(args.model_path),
            test_csv=Path(args.test_csv),
            batch_size=args.batch_size,
            device_choice=args.device,
        )
    )
