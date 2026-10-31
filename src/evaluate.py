"""PyTorch Test Dataset Evaluation Script for Skin Lesion Classification.

Evaluates the trained PyTorch CNN model against the untouched test set (data/processed/test.csv),
computes classification metrics (Accuracy, Precision, Recall, F1 Score, Classification Report),
saves outputs/evaluation_results.json, and generates outputs/confusion_matrix.png.

Runnable via:
    python -m src.evaluate [--batch-size N] [--device auto|cpu|mps] [--model-path PATH] [--test-csv PATH]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_pipeline import LABEL_MAP, NUM_CLASSES, create_dataloader
from src.model import build_model

# Target class order and readable display names
TARGET_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FULL_CLASS_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation script."""
    parser = argparse.ArgumentParser(
        description="PyTorch Test Dataset Evaluation for Skin Lesion Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for test DataLoader")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Hardware execution device selection ('auto', 'cpu', 'mps')",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/skin_lesion_cnn.pth"),
        help="Path to trained PyTorch model weights checkpoint",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/processed/test.csv"),
        help="Path to test split manifest CSV file",
    )
    return parser.parse_args()


def configure_device(device_choice: str) -> Tuple[torch.device, str]:
    """Configure PyTorch device and validate MPS availability."""
    mps_available = torch.backends.mps.is_available()

    if device_choice == "cpu":
        return torch.device("cpu"), "CPU"
    elif device_choice == "mps":
        if not mps_available:
            print("\n========================================")
            print("ERROR: MPS GPU evaluation was requested but MPS is not available.")
            print("========================================\n")
            sys.exit(1)
        return torch.device("mps"), "MPS (Apple Silicon GPU)"
    else:  # auto
        if mps_available:
            return torch.device("mps"), "MPS (Apple Silicon GPU)"
        else:
            return torch.device("cpu"), "CPU"


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """Plot 7x7 confusion matrix using Matplotlib and save image."""
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Skin Lesion CNN Test Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(TARGET_CLASSES))
    plt.xticks(tick_marks, TARGET_CLASSES, rotation=45, fontsize=10)
    plt.yticks(tick_marks, TARGET_CLASSES, fontsize=10)

    # Annotate matrix cells with values
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:,}", horizontalalignment="center", verticalalignment="center", color=color, fontsize=10)

    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_main(args: argparse.Namespace) -> None:
    """Execute model evaluation on test dataset."""
    # 1. File Protection Checks
    if not args.model_path.exists():
        print("\n========================================")
        print(f"[ERROR] Trained model file missing at:\n  {args.model_path}")
        print("Please complete Phase 5 model training first using:")
        print("  python -m src.train --epochs 20 --device mps")
        print("========================================\n")
        sys.exit(1)

    if not args.test_csv.exists():
        print("\n========================================")
        print(f"[ERROR] Test split CSV missing at:\n  {args.test_csv}")
        print("Please complete Phase 3 dataset splitting first using:")
        print("  python -m src.split_dataset")
        print("========================================\n")
        sys.exit(1)

    # 2. Hardware Device Configuration
    device, device_name = configure_device(args.device)

    test_df = pd.read_csv(args.test_csv)
    test_count = len(test_df)

    print("========================================")
    print("SKIN LESION CNN TEST EVALUATION")
    print("========================================")
    print(f"Test Images:      {test_count:,}")
    print(f"Classes:          {NUM_CLASSES}")
    print(f"Device:           {device_name}")
    print(f"\nLoading Model:\n  {args.model_path}")
    print("========================================\n")

    # 3. Load Model and DataLoader
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = create_dataloader(args.test_csv, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    # 4. Evaluation Loop
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. Compute Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec_weighted = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    rec_weighted = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=TARGET_CLASSES,
        zero_division=0,
    )

    cm = confusion_matrix(all_targets, all_preds)

    print("========================================")
    print("TEST RESULTS")
    print("========================================")
    print(f"Accuracy:           {acc * 100.0:.2f}%")
    print(f"Weighted Precision: {prec_weighted * 100.0:.2f}%")
    print(f"Weighted Recall:    {rec_weighted * 100.0:.2f}%")
    print(f"Weighted F1 Score:  {f1_weighted * 100.0:.2f}%")
    print("========================================\n")

    print("Classification Report:")
    print(report_text)

    # 6. Save Evaluation Outputs
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    json_path = outputs_dir / "evaluation_results.json"
    results_data = {
        "accuracy": float(acc),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1_score": float(f1_weighted),
        "test_samples": int(test_count),
        "class_mapping": LABEL_MAP,
        "classification_report": report_dict,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    cm_path = outputs_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path)

    print("========================================")
    print("EVALUATION ARTIFACTS SAVED")
    print("========================================")
    print(f"JSON Results:      {json_path}")
    print(f"Confusion Matrix:  {cm_path}")
    print("========================================\n")


def main() -> None:
    """CLI Entry Point."""
    args = parse_args()
    evaluate_main(args)


if __name__ == "__main__":
    main()
