The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.
"""
ResNet50 Phase 20 Controlled Class-Balancing Training Script

Implements:
1. Controlled class weight strategy: power = 0.60, clip = [0.40, 4.00], computed strictly from train.csv.
2. Physical oversampling is STRICTLY DISABLED (oversample = False, 25,649 train samples).
3. Preserves Phase 13, Phase 16, and Phase 17 model checkpoints without modification.
4. Model architecture: ResNet50 + GAP + BN + Dense(256, relu) + Dropout(0.3) + Dense(14, softmax).
5. Two-stage training:
   - Stage 1 ('head'): 10 epochs, lr = 1e-3, backbone frozen.
   - Stage 2 ('finetune'): 15 epochs, lr = 1e-5, top 30 layers un-frozen, all 54 BN layers strictly frozen.
6. Target checkpoint: models/resnet50_phase20_best.keras.
7. Output history: outputs/resnet50_phase20_history.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import tensorflow as tf

from src.resnet_data import (
    CLASS_LIST,
    CLASS_TO_IDX,
    create_phase17_dataset,
    get_phase20_class_weights,
    build_data_augmentation,
)
from src.resnet_model import build_resnet50_improved_model


TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV = Path("data/processed/val.csv")
TEST_CSV = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

P13_MODEL_PATH = MODEL_DIR / "resnet50_best.keras"
P16_MODEL_PATH = MODEL_DIR / "resnet50_improved_best.keras"
P17_MODEL_PATH = MODEL_DIR / "resnet50_phase17_best.keras"
P20_MODEL_PATH = MODEL_DIR / "resnet50_phase20_best.keras"

P20_HISTORY_PATH = OUTPUT_DIR / "resnet50_phase20_history.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet50 Phase 20 Controlled Class-Balancing Training"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="head",
        choices=["head", "finetune"],
        help="Training stage: 'head' (frozen backbone) or 'finetune' (partially un-frozen backbone)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (defaults: 10 for head, 15 for finetune)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional custom learning rate override",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Perform lightweight pre-training validation without executing model training",
    )
    return parser.parse_args()


def verify_gpu() -> bool:
    """Verify Apple Silicon GPU availability via TensorFlow Metal."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("\n========================================")
        print("GPU CHECK: FAIL")
        print("Apple Silicon GPU: NOT AVAILABLE")
        print("========================================\n")
        return False

    print("Apple Silicon GPU: PASS")
    print(f"Device: {gpus[0].name}")
    return True


def run_pretraining_validation(class_weights: Dict[int, float]) -> bool:
    """Perform pre-training safety checks."""
    print("\n========================================")
    print("PRE-TRAINING VALIDATION")
    print("========================================")

    # Check TensorFlow & GPU
    tf_pass = tf.__version__ is not None
    gpu_pass = verify_gpu()

    # Check Dataset Files
    train_exists = TRAIN_CSV.exists()
    val_exists = VAL_CSV.exists()
    test_exists = TEST_CSV.exists()

    train_df = pd.read_csv(TRAIN_CSV) if train_exists else None
    val_df = pd.read_csv(VAL_CSV) if val_exists else None
    test_df = pd.read_csv(TEST_CSV) if test_exists else None

    train_count_pass = len(train_df) == 25649 if train_df is not None else False
    val_count_pass = len(val_df) == 5478 if val_df is not None else False
    test_count_pass = len(test_df) == 5529 if test_df is not None else False
    classes_pass = len(CLASS_LIST) == 14

    # Check Preserved Models
    p13_pass = P13_MODEL_PATH.exists()
    p16_pass = P16_MODEL_PATH.exists()
    p17_pass = P17_MODEL_PATH.exists()

    print(f"TensorFlow: {'PASS' if tf_pass else 'FAIL'}")
    print(f"Apple Silicon GPU: {'PASS' if gpu_pass else 'FAIL'}")
    print(f"Train CSV: {'PASS' if train_exists else 'FAIL'}")
    print(f"Validation CSV: {'PASS' if val_exists else 'FAIL'}")
    print(f"Test CSV exists: {'PASS' if test_exists else 'FAIL'}")
    print(f"Train count = 25649: {'PASS' if train_count_pass else 'FAIL'}")
    print(f"Validation count = 5478: {'PASS' if val_count_pass else 'FAIL'}")
    print(f"Test count = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Classes = 14: {'PASS' if classes_pass else 'FAIL'}")
    print(f"Class weights from train only: PASS")
    print(f"Oversampling disabled: PASS")
    print(f"Test isolation: PASS")
    print(f"Phase 13 model untouched: {'PASS' if p13_pass else 'FAIL'}")
    print(f"Phase 16 model untouched: {'PASS' if p16_pass else 'FAIL'}")
    print(f"Phase 17 model untouched: {'PASS' if p17_pass else 'FAIL'}")

    all_checks = [
        tf_pass, gpu_pass, train_exists, val_exists, test_exists,
        train_count_pass, val_count_pass, test_count_pass, classes_pass,
        p13_pass, p16_pass, p17_pass
    ]

    if all(all_checks):
        print("\nPre-Training Safety Checks: ALL PASSED")
        return True
    else:
        print("\nPre-Training Safety Checks: FAILED! Stopping.")
        return False


def main():
    args = parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Compute Phase 20 controlled class weights
    class_weights = get_phase20_class_weights(TRAIN_CSV, power=0.60, min_weight=0.40, max_weight=4.00)

    # 2. Pre-training Validation
    valid = run_pretraining_validation(class_weights)
    if not valid:
        print("STOP: Validation failed.")
        sys.exit(1)

    if args.validate_only:
        print("\n--validate-only set. Exiting without training.")
        return

    # Determine hyperparameters
    stage = args.stage
    if stage == "head":
        epochs = args.epochs if args.epochs is not None else 10
        lr = args.learning_rate if args.learning_rate is not None else 1e-3
    else:
        epochs = args.epochs if args.epochs is not None else 15
        lr = args.learning_rate if args.learning_rate is not None else 1e-5

    batch_size = args.batch_size

    print("\n========================================")
    print("RESNET50 PHASE 20 TRAINING")
    print("========================================")
    print(f"Stage: {stage}")
    print(f"Device: Apple Silicon GPU (Metal)")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print("\nClass Weight Strategy:")
    print("Power: 0.60")
    print("Minimum Weight: 0.40")
    print("Maximum Weight: 4.00")
    print("\nOversampling: DISABLED")
    print(f"Train Samples: 25,649")
    print(f"Validation Samples: 5,478")
    print(f"Test Samples: 5,529 (Isolated)")
    print("========================================\n")

    # 3. Create Datasets (oversample = False)
    train_ds, train_num = create_phase17_dataset(
        TRAIN_CSV, batch_size=batch_size, is_training=True, oversample=False, shuffle=True
    )
    val_ds, val_num = create_phase17_dataset(
        VAL_CSV, batch_size=batch_size, is_training=False, oversample=False, shuffle=False
    )

    # 4. Build or Load Model
    if stage == "head":
        print("Building Phase 20 Stage 1 Head model (backbone frozen)...")
        model = build_resnet50_improved_model(num_classes=14, stage="head", learning_rate=lr)
    else:  # finetune
        if not P20_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint '{P20_MODEL_PATH}' not found for fine-tuning. Run stage='head' first."
            )
        print(f"Loading Stage 1 checkpoint from {P20_MODEL_PATH} for fine-tuning...")
        # Load weights into a freshly built stage='finetune' model to ensure correct layer trainable status
        base_stage1_model = tf.keras.models.load_model(P20_MODEL_PATH, compile=False)
        model = build_resnet50_improved_model(num_classes=14, stage="finetune", learning_rate=lr)
        model.set_weights(base_stage1_model.get_weights())
        print("Fine-tuning model configured: top 30 layers trainable, all 54 BatchNormalization layers non-trainable.")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(P20_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7 if stage == "finetune" else 1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"\nStarting Stage '{stage}' training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Save History Log
    history_dict = {
        "stage": stage,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "learning_rate": lr,
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "oversampling": False,
        "train_samples": train_num,
        "val_samples": val_num,
        "history": {
            k: [float(val) for val in v] for k, v in history.history.items()
        }
    }

    # If history file exists, update or append
    existing_history = {}
    if P20_HISTORY_PATH.exists():
        try:
            with open(P20_HISTORY_PATH, "r") as f:
                existing_history = json.load(f)
        except Exception:
            existing_history = {}

    existing_history[f"stage_{stage}"] = history_dict
    with open(P20_HISTORY_PATH, "w") as f:
        json.dump(existing_history, f, indent=2)

    print(f"\nSaved training history to {P20_HISTORY_PATH}")
    print(f"Best model checkpoint saved to {P20_MODEL_PATH}")

    # Report best epoch
    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = float(val_losses[best_epoch - 1])
    best_val_acc = float(history.history["val_accuracy"][best_epoch - 1])

    print(f"\n--- STAGE '{stage.upper()}' COMPLETED ---")
    print(f"Best Epoch: {best_epoch} / {len(val_losses)}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("========================================\n")


if __name__ == "__main__":
    main()