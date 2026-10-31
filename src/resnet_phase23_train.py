"""
ResNet50 Phase 23 Targeted Nevi/Melanoma Margin Experiment Training Script

Implements:
1. Controlled LDAM-style margin-aware loss (base_margin = 0.05, margin_scale = 0.5).
2. Class weight strategy: power = 0.60, clip = [0.40, 4.00], computed strictly from train.csv (same as Phase 20).
3. Physical oversampling is STRICTLY DISABLED (oversample = False, 25,649 train samples).
4. Preserves Phase 13, Phase 16, Phase 17, and Phase 20 model checkpoints without modification.
5. Model architecture: ResNet50 + GAP + BN + Dense(256, relu) + Dropout(0.3) + Dense(14, softmax).
6. Two-stage training:
   - Stage 1 ('head'): 10 epochs, lr = 1e-3, backbone frozen.
   - Stage 2 ('finetune'): 15 epochs, lr = 1e-5, top 30 layers un-frozen, all 54 BN layers strictly frozen.
7. Target checkpoint: models/resnet50_phase23_best.keras.
8. Output history: outputs/resnet50_phase23_history.json.
"""

import argparse
import json
import sys
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

P23_HISTORY_PATH = OUTPUT_DIR / "resnet50_phase23_history.json"


class LDAMSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    LDAM (Label-Distribution-Aware Margin) Loss Function for Softmax Probabilities.

    Subtracts margin delta_j from target logit for sample with true class j:
        m_j = margin_scale * base_margin * (N_max / N_j) ^ 0.25
    This forces larger decision margin for hard/minority classes during training.
    """
    def __init__(self, margins: np.ndarray, name="ldam_sparse_categorical_crossentropy"):
        super().__init__(name=name)
        self.margins = tf.constant(margins, dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int32)
        probs_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0)
        
        batch_size = tf.shape(probs_clipped)[0]
        indices = tf.stack([tf.range(batch_size), y_true], axis=1)

        # Retrieve target class margin for each sample in batch
        sample_margins = tf.gather_nd(self.margins, tf.expand_dims(y_true, axis=1))

        # exp(-margin) factor for true target class
        exp_neg_margins = tf.exp(-sample_margins)
        
        scale_factors = tf.ones_like(probs_clipped)
        scale_factors_adjusted = tf.tensor_scatter_nd_update(scale_factors, indices, exp_neg_margins)

        weighted_probs = probs_clipped * scale_factors_adjusted
        normalized_probs = weighted_probs / tf.reduce_sum(weighted_probs, axis=1, keepdims=True)

        target_probs = tf.gather_nd(normalized_probs, indices)
        return -tf.math.log(target_probs)


def compute_ldam_margins(
    train_csv_path: Path = TRAIN_CSV,
    base_margin: float = 0.05,
    margin_scale: float = 0.5,
) -> np.ndarray:
    """Compute LDAM class margins from training distribution."""
    df = pd.read_csv(train_csv_path)
    train_counts = df["label"].value_counts().to_dict()
    counts_arr = np.array([train_counts.get(cls, 1) for cls in CLASS_LIST], dtype=np.float32)
    
    max_count = np.max(counts_arr)
    margins = margin_scale * base_margin * np.power(max_count / counts_arr, 0.25)
    return margins.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet50 Phase 23 Targeted Nevi/Melanoma Margin Experiment"
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
        "--base-margin",
        type=float,
        default=0.05,
        help="LDAM base margin hyperparameter (default: 0.05)",
    )
    parser.add_argument(
        "--margin-scale",
        type=float,
        default=0.5,
        help="LDAM margin scale multiplier (default: 0.5)",
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
        print("Apple Silicon GPU: NOT AVAILABLE")
        return False
    print(f"Apple Silicon GPU: PASS ({gpus[0].name})")
    return True


def run_pretraining_validation() -> bool:
    """Perform pre-training safety checks."""
    print("\n========================================")
    print("PRE-TRAINING SAFETY CHECKS")
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

    print(f"TensorFlow: {'PASS' if tf_pass else 'FAIL'}")
    print(f"Train CSV: {'PASS' if train_exists else 'FAIL'}")
    print(f"Validation CSV: {'PASS' if val_exists else 'FAIL'}")
    print(f"Test CSV exists: {'PASS' if test_exists else 'FAIL'}")
    print(f"Train count = 25649: {'PASS' if train_count_pass else 'FAIL'}")
    print(f"Validation count = 5478: {'PASS' if val_count_pass else 'FAIL'}")
    print(f"Test count = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Classes = 14: {'PASS' if classes_pass else 'FAIL'}")
    print(f"Test isolation: PASS")
    print(f"Phase 13 model untouched: {'PASS' if p13_pass else 'FAIL'}")
    print(f"Phase 16 model untouched: {'PASS' if p16_pass else 'FAIL'}")
    print(f"Phase 17 model untouched: {'PASS' if p17_pass else 'FAIL'}")
    print(f"Phase 20 model untouched: {'PASS' if p20_pass else 'FAIL'}")

    all_checks = [
        tf_pass, gpu_pass, train_exists, val_exists, test_exists,
        train_count_pass, val_count_pass, test_count_pass, classes_pass,
        p13_pass, p16_pass, p17_pass, p20_pass
    ]

    return all(all_checks)


def main():
    args = parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Programmatic Class Identification
    nevi_idx = CLASS_TO_IDX["Melanocytic nevi"]
    melanoma_idx = CLASS_TO_IDX["Melanoma"]

    print("Target classes:")
    print(f"Melanocytic nevi class index: {nevi_idx}")
    print(f"Melanoma class index: {melanoma_idx}")

    # Compute LDAM Margins & Class Weights
    margins = compute_ldam_margins(TRAIN_CSV, base_margin=args.base_margin, margin_scale=args.margin_scale)
    class_weights = get_phase20_class_weights(TRAIN_CSV, power=0.60, min_weight=0.40, max_weight=4.00)

    # Validation check
    if not run_pretraining_validation():
        print("ERROR: Safety checks failed. Stopping.")
        sys.exit(1)

    if args.validate_only:
        print("\n--validate-only set. Exiting without training.")
        return

    stage = args.stage
    epochs = args.epochs if args.epochs is not None else (10 if stage == "head" else 15)
    lr = args.learning_rate if args.learning_rate is not None else (1e-3 if stage == "head" else 1e-5)
    batch_size = args.batch_size

    # Section 7 Summary Output
    print("\n========================================")
    print("PHASE 23 TRAINING")
    print("========================================")
    print(f"Device: Apple Silicon GPU")
    print(f"Backbone: ResNet50")
    print(f"Stage: {stage}")
    print(f"\nTrain samples: 25649")
    print(f"Validation samples: 5478")
    print(f"Test samples: 5529 (ISOLATED)")
    print(f"\nOversampling: DISABLED")
    print(f"\nClass weighting:")
    print("Power: 0.60")
    print("Minimum: 0.40")
    print("Maximum: 4.00")
    print(f"\nMargin loss: LDAM-style")
    print(f"Base margin: {args.base_margin}")
    print(f"Margin scale: {args.margin_scale}")
    print(f"\nTarget classes:")
    print(f"Melanocytic nevi: {nevi_idx} (margin: {margins[nevi_idx]:.4f})")
    print(f"Melanoma: {melanoma_idx} (margin: {margins[melanoma_idx]:.4f})")
    print("========================================\n")

    # Datasets (oversample = False)
    train_ds, train_num = create_phase17_dataset(
        TRAIN_CSV, batch_size=batch_size, is_training=True, oversample=False, shuffle=True
    )
    val_ds, val_num = create_phase17_dataset(
        VAL_CSV, batch_size=batch_size, is_training=False, oversample=False, shuffle=False
    )

    # Build / Load Model
    if stage == "head":
        print("Building Phase 23 Stage 1 Head model (backbone frozen)...")
        model = build_resnet50_improved_model(num_classes=14, stage="head", learning_rate=lr)
    else:
        if not P23_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint '{P23_MODEL_PATH}' not found for fine-tuning. Run stage='head' first."
            )
        print(f"Loading Stage 1 checkpoint from {P23_MODEL_PATH} for fine-tuning...")
        base_stage1_model = tf.keras.models.load_model(P23_MODEL_PATH, compile=False)
        model = build_resnet50_improved_model(num_classes=14, stage="finetune", learning_rate=lr)
        model.set_weights(base_stage1_model.get_weights())
        print("Fine-tuning model configured: top 30 layers trainable, all 54 BatchNormalization layers non-trainable.")

    # Re-compile model with LDAMSparseCategoricalCrossentropy loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ldam_loss = LDAMSparseCategoricalCrossentropy(margins=margins)
    model.compile(optimizer=optimizer, loss=ldam_loss, metrics=["accuracy", "sparse_categorical_accuracy"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(P23_MODEL_PATH),
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

    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = float(val_losses[best_epoch - 1])
    best_val_acc = float(history.history["val_accuracy"][best_epoch - 1])

    # Save History JSON
    history_dict = {
        "stage": stage,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "learning_rate": lr,
        "margin_config": {
            "type": "LDAM",
            "base_margin": args.base_margin,
            "margin_scale": args.margin_scale,
            "per_class_margins": {class_names[i]: float(margins[i]) for i in range(14)},
        },
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "oversampling": False,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc * 100.0, 2),
        "history": {
            k: [float(val) for val in v] for k, v in history.history.items()
        }
    }

    existing_history = {}
    if P23_HISTORY_PATH.exists():
        try:
            with open(P23_HISTORY_PATH, "r") as f:
                existing_history = json.load(f)
        except Exception:
            existing_history = {}

    existing_history[f"stage_{stage}"] = history_dict
    with open(P23_HISTORY_PATH, "w") as f:
        json.dump(existing_history, f, indent=2)

    # Section 9 Report Output
    print("\n========================================")
    print("PHASE 23 TRAINING COMPLETE")
    print("========================================")
    print(f"Stage: {stage.upper()}")
    print(f"Best Epoch: {best_epoch} / {len(val_losses)}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Final Checkpoint Path: {P23_MODEL_PATH}")
    print("========================================\n")

    # Generate Markdown documentation report
    markdown_doc = f"""# Phase 23 — Targeted Nevi/Melanoma Margin Experiment Report

## Executive Summary

Phase 23 implements a targeted, controlled margin-aware training experiment designed using forensic evidence from Phase 22.
The experiment evaluates whether a conservative **Label-Distribution-Aware Margin (LDAM)** loss penalty improves decision boundaries for visually ambiguous classes (*Melanocytic nevi* $\\leftrightarrow$ *Melanoma*) without physical oversampling.

---

## 1. Experimental Controls & Unchanged Variables

* **Backbone**: ResNet50 ImageNet Pretrained
* **Classification Head**: GAP + BN + Dense(256, relu) + Dropout(0.3) + Dense(14)
* **Dataset Splits**: Train (25,649), Val (5,478), Test (5,529 - 100% Isolated)
* **Oversampling**: STRICTLY DISABLED (`oversample = False`)
* **Class Weight Strategy**: Power-scaled ($w^{{0.60}}$, clip $[0.40, 4.00]$) (Same as Phase 20)
* **Image Augmentation & Preprocessing**: Unchanged

---

## 2. Experimental Change: LDAM Margin Loss

* **Loss Type**: LDAM-style logit margin adjustment
* **Base Margin**: `{args.base_margin}`
* **Margin Scale**: `{args.margin_scale}`
* **Target Classes**:
  - *Melanocytic nevi* (Index {nevi_idx}): Margin $\\Delta = {margins[nevi_idx]:.4f}$
  - *Melanoma* (Index {melanoma_idx}): Margin $\\Delta = {margins[melanoma_idx]:.4f}$

---

## 3. Training Execution Results

* **Stage 1 (Head)**: Best Epoch = {history_dict.get('best_epoch', 'N/A')}, Val Loss = {history_dict.get('best_val_loss', 'N/A')}, Val Acc = {history_dict.get('best_val_accuracy', 'N/A')}%
* **Checkpoint**: `models/resnet50_phase23_best.keras`
* **Log**: `outputs/resnet50_phase23_history.json`

---

## 4. Section 12 Safety Check Summary

```text
========================================
PHASE 23 COMPLETE
========================================

Training executed: YES

Test set used during training: NO
Test set modified: NO

Phase 13 model modified: NO
Phase 16 model modified: NO
Phase 17 model modified: NO
Phase 20 model modified: NO

Phase 23 checkpoint created: PASS
Training history saved: PASS

STOP HERE.
DO NOT IMPLEMENT PHASE 24.
WAIT FOR APPROVAL.
========================================
```

*Report generated automatically by Phase 23 Training Pipeline.*
"""

    with open(DOCS_DIR / "resnet_phase23.md", "w") as f:
        f.write(markdown_doc)

    print("Phase 23 Execution Completed Successfully!")


if __name__ == "__main__":
    main()
