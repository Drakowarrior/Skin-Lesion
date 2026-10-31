"""
ResNet50 Test Evaluation & Top-3 Prediction Analysis (Phase 13)

Evaluates the trained ResNet50 model (models/resnet50_best.keras) on the untouched
test dataset (data/processed/test.csv).

Computes:
1. Top-1 and Top-3 accuracy & improvement.
2. Macro and Weighted Precision, Recall, and F1 scores.
3. Per-class metrics (Precision, Recall, F1, Support, Top-1 Acc, Top-3 Acc).
4. Prediction probability / confidence distribution analysis.
5. Top-3 error transition analysis (Top-1 incorrect -> Top-3 correct).
6. Confusion matrix & normalized confusion matrix visual plots.
7. Machine-readable JSON summary and per-image prediction outputs.
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import tensorflow as tf

from src.resnet_data import CLASS_LIST, CLASS_TO_IDX, IDX_TO_CLASS, create_dataset


DEFAULT_MODEL_PATH = Path("models/resnet50_best.keras")
DEFAULT_TEST_CSV = Path("data/processed/test.csv")
OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet50 Test Evaluation & Top-3 Prediction Analysis"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained ResNet50 model checkpoint (.keras)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(DEFAULT_TEST_CSV),
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


def evaluate_resnet50(
    model_path: Path,
    test_csv: Path,
    batch_size: int = 32,
    device_choice: str = "auto",
):
    """Execute complete ResNet50 evaluation and generate all Phase 13 artifacts."""
    device_desc = setup_device(device_choice)

    if not model_path.exists():
        print(f"ERROR: Trained model checkpoint '{model_path}' not found.")
        sys.exit(1)

    if not test_csv.exists():
        print(f"ERROR: Test split CSV '{test_csv}' not found.")
        sys.exit(1)

    # Read Test CSV metadata
    test_df = pd.read_csv(test_csv)
    total_test_images = len(test_df)
    image_paths_list = test_df["image_path"].tolist()

    print("========================================")
    print("RESNET50 TEST EVALUATION")
    print("========================================")
    print(f"\nTensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {getattr(tf.keras, '__version__', '2.x/3.x')}")
    print(f"GPU Available: {'YES' if 'GPU' in device_desc else 'NO'}")
    print(f"Device: {device_desc}")
    print(f"\nModel: {model_path}")
    print(f"Test Images: {total_test_images}")
    print(f"Classes: {len(CLASS_LIST)}")
    print("========================================\n")

    # Load Model & Dataset
    print("Loading ResNet50 model checkpoint...")
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

    # Top-1 Predictions
    y_pred_top1 = np.argmax(y_probs, axis=1)
    top1_probs = np.max(y_probs, axis=1)

    # Top-3 Predictions & Correctness Analysis
    top3_indices = np.argsort(y_probs, axis=1)[:, -3:][:, ::-1]  # Shape: (N, 3)
    
    top1_correct = (y_pred_top1 == y_true)
    top3_correct = np.array([y_true[i] in top3_indices[i] for i in range(total_test_images)])

    # Metric Calculations
    top1_acc = np.mean(top1_correct) * 100.0
    top3_acc = np.mean(top3_correct) * 100.0
    top3_improvement = top3_acc - top1_acc

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred_top1, average="macro")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred_top1, average="weighted")

    macro_p *= 100.0
    macro_r *= 100.0
    macro_f1 *= 100.0
    weighted_p *= 100.0
    weighted_r *= 100.0
    weighted_f1 *= 100.0

    # Confidence Analysis
    avg_top1_prob = float(np.mean(top1_probs) * 100.0)
    median_top1_prob = float(np.median(top1_probs) * 100.0)
    min_top1_prob = float(np.min(top1_probs) * 100.0)
    max_top1_prob = float(np.max(top1_probs) * 100.0)

    correct_prob = float(np.mean(top1_probs[top1_correct]) * 100.0) if np.any(top1_correct) else 0.0
    incorrect_prob = float(np.mean(top1_probs[~top1_correct]) * 100.0) if np.any(~top1_correct) else 0.0

    low_prob_50 = int(np.sum(top1_probs < 0.50))
    low_prob_70 = int(np.sum(top1_probs < 0.70))
    low_prob_90 = int(np.sum(top1_probs < 0.90))

    # Top-3 Transition Error Analysis
    top1_inc_top3_cor = int(np.sum(~top1_correct & top3_correct))
    top1_inc_top3_cor_pct = float(top1_inc_top3_cor / total_test_images * 100.0)

    # Per-Class Metrics Calculation
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred_top1, labels=np.arange(len(CLASS_LIST))
    )

    per_class_metrics = {}
    print("\n----------------------------------------")
    print("PER-CLASS PERFORMANCE & TOP-3 RECOGNITION")
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

    cm_path = OUTPUT_DIR / "resnet50_confusion_matrix.png"
    cm_norm_path = OUTPUT_DIR / "resnet50_confusion_matrix_normalized.png"

    plot_confusion_matrix(cm, CLASS_LIST, cm_path, "ResNet50 Confusion Matrix (Raw Counts)", fmt="d")
    plot_confusion_matrix(cm_norm, CLASS_LIST, cm_norm_path, "ResNet50 Normalized Confusion Matrix (%)", fmt=".1%", cmap=plt.cm.Greens)

    print(f"\nSaved raw confusion matrix: {cm_path}")
    print(f"Saved normalized confusion matrix: {cm_norm_path}")

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

    pred_json_path = OUTPUT_DIR / "resnet50_predictions.json"
    with open(pred_json_path, "w") as f:
        json.dump(predictions_records, f, indent=2)
    print(f"Saved detailed predictions JSON: {pred_json_path}")

    # Generate Aggregated Summary JSON
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
            "below_50_probability_count": low_prob_50,
            "below_70_probability_count": low_prob_70,
            "below_90_probability_count": low_prob_90,
        },
        "top3_analysis": {
            "top1_incorrect_top3_correct_count": top1_inc_top3_cor,
            "top1_incorrect_top3_correct_percentage": round(top1_inc_top3_cor_pct, 2),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix_paths": {
            "raw": str(cm_path),
            "normalized": str(cm_norm_path),
        },
    }

    eval_json_path = OUTPUT_DIR / "resnet50_evaluation_results.json"
    with open(eval_json_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Saved evaluation summary JSON: {eval_json_path}")

    # Print Final Summary Block
    print("\n========================================")
    print("TEST RESULTS")
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
    print(f"\nCorrect Prediction Probability: {correct_prob:.2f}%")
    print(f"Incorrect Prediction Probability: {incorrect_prob:.2f}%")
    print(f"\n<50% Probability: {low_prob_50} / {total_test_images}")
    print(f"<70% Probability: {low_prob_70} / {total_test_images}")
    print(f"<90% Probability: {low_prob_90} / {total_test_images}")

    print("\n========================================")
    print("TOP-3 ANALYSIS")
    print("========================================")
    print(f"Top-1 Incorrect → Top-3 Correct:")
    print(f"{top1_inc_top3_cor} / {total_test_images} ({top1_inc_top3_cor_pct:.2f}%)")

    print("\n========================================")
    print("ARTIFACTS")
    print("========================================")
    print(f"Evaluation JSON:\n  {eval_json_path}")
    print(f"\nPredictions JSON:\n  {pred_json_path}")
    print(f"\nConfusion Matrix:\n  {cm_path}")
    print(f"\nNormalized Confusion Matrix:\n  {cm_norm_path}")

    print("\n========================================")
    print("STATUS: PASS")
    print("========================================")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        evaluate_resnet50(
            model_path=Path(args.model_path),
            test_csv=Path(args.test_csv),
            batch_size=args.batch_size,
            device_choice=args.device,
        )
    )
