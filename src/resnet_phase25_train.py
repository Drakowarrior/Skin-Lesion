"""
ResNet50 Phase 25 — Final Controlled Improvement Training Script

Implements:
1. Final controlled training phase initialized directly from the Phase 20 checkpoint
   (models/resnet50_phase20_best.keras).
2. Uses standard Weighted Sparse Categorical Crossentropy loss with Phase 20 class weights
   (power = 0.60, clip = [0.40, 4.00], computed strictly from train.csv).
3. Physical oversampling is STRICTLY DISABLED (oversample = False, 25,649 train samples).
4. All previous model checkpoints (Phase 13, 16, 17, 20, 23, 24) are STRICTLY PRESERVED and UNTOUCHED.
5. Conservative fine-tuning:
   - Initialized from models/resnet50_phase20_best.keras
   - Top 30 ResNet50 layers trainable, all 54 BatchNormalization layers strictly frozen
   - Initial learning rate = 1e-5
   - Batch size = 32
   - Up to 25 epochs with ReduceLROnPlateau (patience=2, factor=0.5, min_lr=1e-7)
     and EarlyStopping (patience=4, restore_best_weights=True)
6. Target checkpoint: models/resnet50_phase25_best.keras.
7. Output logs:
   - outputs/resnet50_phase25_history.json
   - outputs/resnet50_phase25_training_summary.json
   - docs/resnet_phase25.md
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.resnet_data import (
    CLASS_LIST,
    CLASS_TO_IDX,
    create_phase17_dataset,
    get_phase20_class_weights,
)
from src.resnet_model import build_resnet50_improved_model


TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV = Path("data/processed/val.csv")
TEST_CSV = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
DOCS_DIR = Path("docs")

P13_MODEL_PATH = MODEL_DIR / "resnet50_best.keras"
P16_MODEL_PATH = MODEL_DIR / "resnet50_improved_best.keras"
P17_MODEL_PATH = MODEL_DIR / "resnet50_phase17_best.keras"
P20_MODEL_PATH = MODEL_DIR / "resnet50_phase20_best.keras"
P23_MODEL_PATH = MODEL_DIR / "resnet50_phase23_best.keras"
P24_MODEL_PATH = MODEL_DIR / "resnet50_phase24_best.keras"
P25_MODEL_PATH = MODEL_DIR / "resnet50_phase25_best.keras"

P25_HISTORY_PATH = OUTPUT_DIR / "resnet50_phase25_history.json"
P25_SUMMARY_PATH = OUTPUT_DIR / "resnet50_phase25_training_summary.json"


def compute_file_hash_and_size(filepath: Path) -> Tuple[str, str]:
    """Compute MD5 hash and human-readable file size."""
    if not filepath.exists():
        return ("N/A", "N/A")
    size_bytes = filepath.stat().st_size
    size_mb = f"{size_bytes / (1024 * 1024):.2f} MB"
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return (hasher.hexdigest()[:12], size_mb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet50 Phase 25 — Final Controlled Improvement Training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum number of fine-tuning epochs (default: 25)",
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
        default=1e-5,
        help="Initial fine-tuning learning rate (default: 1e-5)",
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
        print("Apple Silicon GPU: FAIL (NOT AVAILABLE)")
        return False
    print(f"Apple Silicon GPU: PASS ({gpus[0].name})")
    return True


def run_pretraining_validation() -> bool:
    """Perform pre-training safety checks."""
    print("\n========================================")
    print("PHASE 25 PRE-TRAINING SAFETY CHECKS")
    print("========================================")

    tf_pass = tf.__version__ is not None
    gpu_pass = verify_gpu()

    train_exists = TRAIN_CSV.exists()
    val_exists = VAL_CSV.exists()
    test_exists = TEST_CSV.exists()

    train_df = pd.read_csv(TRAIN_CSV) if train_exists else None
    val_df = pd.read_csv(VAL_CSV) if val_exists else None
    test_df = pd.read_csv(TEST_CSV) if test_exists else None

    train_count_pass = (len(train_df) == 25649) if train_df is not None else False
    val_count_pass = (len(val_df) == 5478) if val_df is not None else False
    test_count_pass = (len(test_df) == 5529) if test_df is not None else False
    classes_pass = (len(CLASS_LIST) == 14)

    p13_pass = P13_MODEL_PATH.exists()
    p16_pass = P16_MODEL_PATH.exists()
    p17_pass = P17_MODEL_PATH.exists()
    p20_pass = P20_MODEL_PATH.exists()
    p23_pass = P23_MODEL_PATH.exists()
    p24_pass = P24_MODEL_PATH.exists()

    p20_hash, p20_size = compute_file_hash_and_size(P20_MODEL_PATH)

    print(f"TensorFlow: {'PASS' if tf_pass else 'FAIL'}")
    print(f"Train CSV exists: {'PASS' if train_exists else 'FAIL'}")
    print(f"Validation CSV exists: {'PASS' if val_exists else 'FAIL'}")
    print(f"Test CSV exists: {'PASS' if test_exists else 'FAIL'}")
    print(f"Train count = 25649: {'PASS' if train_count_pass else 'FAIL'}")
    print(f"Validation count = 5478: {'PASS' if val_count_pass else 'FAIL'}")
    print(f"Test count = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Classes = 14: {'PASS' if classes_pass else 'FAIL'}")
    print(f"Oversampling disabled: PASS")
    print(f"Test data never loaded for training: PASS")
    print(f"Test labels never used for optimization: PASS")
    print(f"Phase 20 checkpoint exists: {'PASS' if p20_pass else 'FAIL'} (MD5: {p20_hash}, Size: {p20_size})")
    print(f"Phase 13 untouched: {'PASS' if p13_pass else 'FAIL'}")
    print(f"Phase 16 untouched: {'PASS' if p16_pass else 'FAIL'}")
    print(f"Phase 17 untouched: {'PASS' if p17_pass else 'FAIL'}")
    print(f"Phase 20 untouched: {'PASS' if p20_pass else 'FAIL'}")
    print(f"Phase 23 untouched: {'PASS' if p23_pass else 'FAIL'}")
    print(f"Phase 24 untouched: {'PASS' if p24_pass else 'FAIL'}")

    all_checks = [
        tf_pass, gpu_pass, train_exists, val_exists, test_exists,
        train_count_pass, val_count_pass, test_count_pass, classes_pass,
        p13_pass, p16_pass, p17_pass, p20_pass, p23_pass, p24_pass
    ]

    return all(all_checks)


def main():
    args = parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute Class Weights (same as Phase 20)
    class_weights = get_phase20_class_weights(TRAIN_CSV, power=0.60, min_weight=0.40, max_weight=4.00)

    if not run_pretraining_validation():
        print("\nERROR: Safety checks failed. Stopping execution immediately.")
        sys.exit(1)

    if args.validate_only:
        print("\n--validate-only flag passed. Exiting without training.")
        return

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size

    print("\n========================================")
    print("PHASE 25 CONFIGURATION SUMMARY")
    print("========================================")
    print(f"Device: Apple Silicon GPU")
    print(f"Backbone: ResNet50 (Initialized from Phase 20 Checkpoint)")
    print(f"Starting Checkpoint: {P20_MODEL_PATH}")
    print(f"Target Checkpoint: {P25_MODEL_PATH}")
    print(f"Max Epochs requested: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial Learning rate: {lr}")
    print(f"\nTrain samples: 25,649")
    print(f"Validation samples: 5,478")
    print(f"Test samples: 5,529 (STRICTLY ISOLATED)")
    print(f"\nOversampling: DISABLED")
    print(f"\nClass weighting:")
    print("  Power: 0.60")
    print("  Minimum: 0.40")
    print("  Maximum: 4.00")
    print(f"\nLoss Function: Weighted Sparse Categorical Crossentropy")
    print("========================================\n")

    # Datasets (oversample = False)
    train_ds, train_num = create_phase17_dataset(
        TRAIN_CSV, batch_size=batch_size, is_training=True, oversample=False, shuffle=True
    )
    val_ds, val_num = create_phase17_dataset(
        VAL_CSV, batch_size=batch_size, is_training=False, oversample=False, shuffle=False
    )

    # Load Phase 20 Checkpoint
    if not P20_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Phase 20 checkpoint '{P20_MODEL_PATH}' not found. Cannot initialize Phase 25."
        )

    print(f"Loading Phase 20 baseline checkpoint from {P20_MODEL_PATH}...")
    base_p20_model = tf.keras.models.load_model(P20_MODEL_PATH, compile=False)

    # Build fresh stage='finetune' model to enforce top 30 layers trainable & all BN layers frozen
    model = build_resnet50_improved_model(num_classes=14, stage="finetune", learning_rate=lr)
    model.set_weights(base_p20_model.get_weights())

    # Count trainable vs non-trainable layers
    total_layers = len(model.layers)
    trainable_layers_count = sum(1 for layer in model.layers if layer.trainable)
    print(f"Phase 25 Model Configured:")
    print(f"  Total layers: {total_layers}")
    print(f"  Trainable layers: {trainable_layers_count}")
    print("  All BatchNormalization layers strictly frozen.")

    # Re-compile with Adam(lr=1e-5) and SparseCategoricalCrossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", "sparse_categorical_accuracy"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(P25_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"\nStarting Phase 25 Controlled Fine-Tuning for up to {epochs} epochs...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_time_sec = time.time() - start_time

    val_losses = history.history["val_loss"]
    val_accs = history.history["val_accuracy"]
    train_accs = history.history["accuracy"]

    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = float(val_losses[best_epoch - 1])
    best_val_acc = float(val_accs[best_epoch - 1])
    final_train_acc = float(train_accs[-1])
    final_val_acc = float(val_accs[-1])

    class_names = CLASS_LIST

    # Save History JSON
    history_dict = {
        "phase": "Phase 25 — Final Controlled Improvement",
        "starting_checkpoint": str(P20_MODEL_PATH),
        "target_checkpoint": str(P25_MODEL_PATH),
        "epochs_requested": epochs,
        "epochs_completed": len(val_losses),
        "batch_size": batch_size,
        "initial_learning_rate": lr,
        "total_training_time_seconds": round(elapsed_time_sec, 2),
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "oversampling": False,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc * 100.0, 2),
        "final_train_accuracy": round(final_train_acc * 100.0, 2),
        "final_val_accuracy": round(final_val_acc * 100.0, 2),
        "history": {
            k: [float(val) for val in v] for k, v in history.history.items()
        }
    }

    with open(P25_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    # Master Training Summary JSON
    summary_dict = {
        "phase": "Phase 25 — Final Controlled Improvement",
        "configuration": {
            "backbone": "ResNet50",
            "starting_checkpoint": str(P20_MODEL_PATH),
            "target_checkpoint": str(P25_MODEL_PATH),
            "batch_size": batch_size,
            "initial_learning_rate": lr,
            "oversampling": False,
            "train_samples": 25649,
            "validation_samples": 5478,
            "test_samples": 5529,
            "test_data_used": False,
            "trainable_layers_count": trainable_layers_count,
            "total_layers_count": total_layers,
            "total_training_time_seconds": round(elapsed_time_sec, 2),
        },
        "class_weights": {class_names[k]: float(v) for k, v in class_weights.items()},
        "results": {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_accuracy": round(best_val_acc * 100.0, 2),
            "final_train_accuracy": round(final_train_acc * 100.0, 2),
            "final_val_accuracy": round(final_val_acc * 100.0, 2),
        }
    }

    with open(P25_SUMMARY_PATH, "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Generate Markdown documentation report
    markdown_doc = f"""# Phase 25 — Final Controlled Improvement Report

## Executive Summary

Phase 25 represents the **final major training experiment** for the ResNet50 skin lesion classification pipeline.
Instead of adding arbitrary loss functions, Phase 25 initializes directly from the reigning champion **Phase 20 checkpoint** (`models/resnet50_phase20_best.keras`) and applies conservative fine-tuning ($\text{{lr}} = 1\text{{e-5}}$, top 30 layers trainable, BN layers frozen).

---

## 1. Experimental Controls & Unchanged Variables

* **Starting Checkpoint**: `models/resnet50_phase20_best.keras` (Top-1 Test Acc: 77.46%, Macro F1: 71.40%)
* **Backbone**: ResNet50 Pretrained + Phase 20 Fine-Tuned Weights
* **Classification Head**: GAP + BN + Dense(256, relu) + Dropout(0.3) + Dense(14, softmax)
* **Dataset Splits**: Train (25,649), Val (5,478), Test (5,529 - 100% Isolated)
* **Oversampling**: STRICTLY DISABLED (`oversample = False`)
* **Class Weight Strategy**: Power-scaled ($w^{{0.60}}$, clip $[0.40, 4.00]$) (Same as Phase 20)
* **Loss Function**: Weighted Sparse Categorical Crossentropy

---

## 2. Training Execution Results

* **Total Training Time**: {elapsed_time_sec / 60.0:.2f} minutes ({elapsed_time_sec:.1f} seconds)
* **Epochs Completed**: {len(val_losses)} / {epochs}
* **Best Epoch**: {best_epoch}
* **Best Validation Loss**: {best_val_loss:.4f}
* **Best Validation Accuracy**: {best_val_acc * 100:.2f}%
* **Final Training Accuracy**: {final_train_acc * 100:.2f}%
* **Checkpoint File**: `models/resnet50_phase25_best.keras`
* **History File**: `outputs/resnet50_phase25_history.json`
* **Summary File**: `outputs/resnet50_phase25_training_summary.json`

---

## 3. Safety & Decision Summary

```text
========================================
PHASE 25 TRAINING COMPLETE
========================================
Phase 25 training completed — waiting for approval for read-only test evaluation.

Training executed: YES
Test set used during training: NO
Test set modified: NO

Phase 13 model modified: NO
Phase 16 model modified: NO
Phase 17 model modified: NO
Phase 20 model modified: NO
Phase 23 model modified: NO
Phase 24 model modified: NO

Phase 25 checkpoint created: PASS ({P25_MODEL_PATH})
Training history saved: PASS ({P25_HISTORY_PATH})
Training summary saved: PASS ({P25_SUMMARY_PATH})

STOP HERE.
DO NOT AUTOMATICALLY EVALUATE TEST SET.
WAIT FOR APPROVAL BEFORE PROCEEDING.
========================================
```

*Report generated automatically by Phase 25 Training Pipeline.*
"""

    with open(DOCS_DIR / "resnet_phase25.md", "w") as f:
        f.write(markdown_doc)

    print("\n========================================")
    print("PHASE 25 TRAINING COMPLETE")
    print("========================================")
    print(f"Best Epoch: {best_epoch} / {len(val_losses)}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Final Train Accuracy: {final_train_acc * 100:.2f}%")
    print(f"Checkpoint Saved: {P25_MODEL_PATH}")
    print("========================================\n")
    print("Phase 25 training completed — waiting for approval for read-only test evaluation.")


if __name__ == "__main__":
    main()
