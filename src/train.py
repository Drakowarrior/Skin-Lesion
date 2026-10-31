"""PyTorch GPU (MPS) CNN Training Pipeline for Skin Lesion Classification.

Executes training from terminal:
    python -m src.train [--epochs N] [--batch-size N] [--learning-rate F] [--device auto|cpu|mps] [--force]

Features:
- PyTorch training loop targeting Mac Apple Silicon MPS GPU acceleration.
- CLI arguments: --epochs, --batch-size, --learning-rate, --device, --force.
- Strict error handling when --device mps is requested but MPS is unavailable.
- Balanced class weight computation using train.csv only.
- PyTorch nn.CrossEntropyLoss and torch.optim.Adam.
- Best model checkpointing to models/skin_lesion_cnn.pth based on val_loss.
- Early stopping (patience = 5 epochs).
- Saves training history to models/training_history.json.
- Post-training validation (reloads saved model and verifies 1 batch prediction shape).
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import sklearn.utils.class_weight as sklearn_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_pipeline import LABEL_MAP, NUM_CLASSES, create_dataloader
from src.model import build_model

RANDOM_SEED: int = 42


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training script."""
    parser = argparse.ArgumentParser(
        description="PyTorch MPS GPU Training Pipeline for Skin Lesion Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial Adam learning rate")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Hardware execution device selection ('auto', 'cpu', 'mps')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing models/skin_lesion_cnn.pth model artifact",
    )
    return parser.parse_args()


def configure_device(device_choice: str) -> Tuple[torch.device, str]:
    """Configure PyTorch device (mps, cpu, auto) and validate availability.

    Args:
        device_choice: Device option string ('auto', 'cpu', 'mps').

    Returns:
        Tuple of (torch.device, device_name_str).
    """
    mps_available = torch.backends.mps.is_available()

    if device_choice == "cpu":
        return torch.device("cpu"), "CPU"

    elif device_choice == "mps":
        if not mps_available:
            print("\n========================================")
            print("ERROR: MPS GPU training was requested but MPS is not available.")
            print("========================================\n")
            sys.exit(1)
        return torch.device("mps"), "MPS (Apple Silicon GPU)"

    else:  # auto
        if mps_available:
            return torch.device("mps"), "MPS (Apple Silicon GPU)"
        else:
            return torch.device("cpu"), "CPU"


def compute_class_weights(train_csv: Path) -> np.ndarray:
    """Compute balanced class weights on train.csv only."""
    df = pd.read_csv(train_csv)
    train_labels = np.array([LABEL_MAP[l] for l in df["label"].values])
    unique_classes = np.unique(train_labels)

    computed_weights = sklearn_class_weight.compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels,
    )

    weights_dict = {int(c): float(w) for c, w in zip(unique_classes, computed_weights)}
    full_weights = [weights_dict.get(c, 1.0) for c in range(NUM_CLASSES)]
    return np.array(full_weights, dtype=np.float32)


def post_training_validation(model_path: Path, val_csv: Path, batch_size: int, device: torch.device) -> None:
    """Load saved model checkpoint and verify 1 batch prediction shape."""
    print("\n========================================")
    print("POST-TRAINING MODEL VALIDATION")
    print("========================================")
    if not model_path.exists():
        print(f"[FAIL] Saved model missing at: {model_path}")
        return

    print(f"Loading saved PyTorch model from:\n  {model_path}")
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    val_loader = create_dataloader(val_csv, batch_size=batch_size, shuffle=False, num_workers=0)
    val_images, val_labels = next(iter(val_loader))
    val_images = val_images.to(device)

    with torch.no_grad():
        logits = model(val_images)

    print(f"\nInput Shape:  {val_images.shape}")
    print(f"Output Logits Shape: {logits.shape}")
    assert logits.shape == (val_images.shape[0], NUM_CLASSES), "Incorrect prediction shape!"
    print("========================================\n")


def train_main(args: argparse.Namespace) -> None:
    """Execute PyTorch training pipeline."""
    set_seed(RANDOM_SEED)

    # 1. Configure Hardware Device
    device, device_name = configure_device(args.device)

    print("\n========================================")
    print("PYTORCH TRAINING DEVICE")
    print("========================================")
    print(f"MPS Available:   {'Yes' if torch.backends.mps.is_available() else 'No'}")
    print(f"Selected Device: {device_name}")
    print("========================================\n")

    # 2. Existing Model Protection
    models_dir = Path("models")
    model_path = models_dir / "skin_lesion_cnn.pth"
    history_path = models_dir / "training_history.json"
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and not args.force:
        print("\nA trained model already exists.\n")
        print("Use --force to overwrite it.")
        print("Example: python -m src.train --force\n")
        sys.exit(0)

    train_csv = Path("data/processed/train.csv")
    val_csv = Path("data/processed/val.csv")

    if not train_csv.exists() or not val_csv.exists():
        print("\n[ERROR] Processed train/val CSV files not found.")
        print("Please complete Phase 3 first using: python -m src.split_dataset\n")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # 3. Class Weight Calculation
    class_weights_arr = compute_class_weights(train_csv)
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32).to(device)

    print("========================================")
    print("CLASS WEIGHTS (Calculated on train.csv)")
    print("========================================")
    rev_map = {v: k for k, v in LABEL_MAP.items()}
    for c_idx in range(NUM_CLASSES):
        print(f"  {rev_map[c_idx]:<6} ({c_idx}): {class_weights_arr[c_idx]:.4f}")

    # 4. Print Startup Information
    print("\n========================================")
    print("SKIN LESION CNN TRAINING")
    print("========================================")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.learning_rate}")
    print(f"Training Images:   {len(train_df):,}")
    print(f"Validation Images: {len(val_df):,}")
    print(f"Classes:          {NUM_CLASSES}")
    print(f"Device:           {device_name}")
    print("========================================\n")

    # 5. DataLoaders, Model, Criterion, Optimizer
    train_loader = create_dataloader(train_csv, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_csv, batch_size=args.batch_size, shuffle=False)

    model = build_model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        print(f"Epoch {epoch:>2d}/{args.epochs}")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")

        # Best Model Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  --> Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (Val loss did not improve for {patience} epochs).")
                break

    # Save History JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")
    print(f"Saved Best Model:   {model_path}")
    print(f"Saved History:      {history_path}")
    print("========================================\n")

    # Post-Training Validation
    post_training_validation(model_path, val_csv, args.batch_size, device)


def main() -> None:
    """CLI Entry Point."""
    args = parse_args()
    train_main(args)


if __name__ == "__main__":
    main()
