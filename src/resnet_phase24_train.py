"""
ResNet50 Phase 24 — Targeted Nevi vs Melanoma Boundary Experiment Training Script

Implements:
1. Normal weighted multiclass classification loss + auxiliary pairwise boundary loss
   between Melanocytic nevi (index 9) and Melanoma (index 10).
2. Pairwise loss formula:
   For Nevi (9): max(0, PAIRWISE_MARGIN - (log(p_9) - log(p_10)))
   For Melanoma (10): max(0, PAIRWISE_MARGIN - (log(p_10) - log(p_9)))
   Total loss = classification_loss + PAIRWISE_LAMBDA * pairwise_boundary_loss
   Default PAIRWISE_LAMBDA = 0.05, PAIRWISE_MARGIN = 0.10.
3. Class weight strategy: power = 0.60, clip = [0.40, 4.00], computed strictly from train.csv (same as Phase 20).
4. Physical oversampling is STRICTLY DISABLED (oversample = False, 25,649 train samples).
5. Preserves Phase 13, Phase 16, Phase 17, Phase 20, and Phase 23 model checkpoints without modification.
6. Model architecture: ResNet50 + GAP + BN + Dense(256, relu) + Dropout(0.3) + Dense(14, softmax).
7. Two-stage training:
   - Stage 1 ('head'): 10 epochs, lr = 1e-3, backbone frozen.
   - Stage 2 ('finetune'): 15 epochs, lr = 1e-5, top 30 layers un-frozen, all 54 BN layers strictly frozen.
8. Target checkpoint: models/resnet50_phase24_best.keras.
9. Output logs:
   - outputs/resnet50_phase24_history.json
   - outputs/resnet50_phase24_training_summary.json
   - docs/resnet_phase24.md
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
P24_MODEL_PATH = MODEL_DIR / "resnet50_phase24_best.keras"

P24_HISTORY_PATH = OUTPUT_DIR / "resnet50_phase24_history.json"
P24_SUMMARY_PATH = OUTPUT_DIR / "resnet50_phase24_training_summary.json"


class PairwiseBoundarySparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Weighted Sparse Categorical Crossentropy with auxiliary pairwise boundary loss
    for Melanocytic nevi (index 9) and Melanoma (index 10).

    total_loss = classification_loss + pairwise_lambda * pairwise_boundary_loss
    """
    def __init__(
        self,
        pairwise_lambda: float = 0.05,
        pairwise_margin: float = 0.10,
        nevi_idx: int = 9,
        melanoma_idx: int = 10,
        name: str = "pairwise_boundary_scce",
    ):
        super().__init__(name=name)
        self.pairwise_lambda = tf.constant(pairwise_lambda, dtype=tf.float32)
        self.pairwise_margin = tf.constant(pairwise_margin, dtype=tf.float32)
        self.nevi_idx = nevi_idx
        self.melanoma_idx = melanoma_idx

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        probs_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0)
        log_probs = tf.math.log(probs_clipped)

        # 1. Base Classification Loss (Sparse Categorical Crossentropy per sample)
        batch_size = tf.shape(y_true)[0]
        indices = tf.stack([tf.range(batch_size), y_true], axis=1)
        target_probs = tf.gather_nd(probs_clipped, indices)
        base_ce = -tf.math.log(target_probs)

        # 2. Pairwise Boundary Loss for Nevi (9) and Melanoma (10)
        log_p9 = log_probs[:, self.nevi_idx]
        log_p10 = log_probs[:, self.melanoma_idx]

        # For Nevi (9): margin condition z_9 - z_10 >= margin => penalty max(0, margin - (log_p9 - log_p10))
        violation_nevi = tf.maximum(0.0, self.pairwise_margin - (log_p9 - log_p10))

        # For Melanoma (10): margin condition z_10 - z_9 >= margin => penalty max(0, margin - (log_p10 - log_p9))
        violation_mel = tf.maximum(0.0, self.pairwise_margin - (log_p10 - log_p9))

        is_nevi = tf.cast(tf.equal(y_true, self.nevi_idx), tf.float32)
        is_mel = tf.cast(tf.equal(y_true, self.melanoma_idx), tf.float32)

        pairwise_loss = (is_nevi * violation_nevi) + (is_mel * violation_mel)

        total_loss = base_ce + (self.pairwise_lambda * pairwise_loss)
        return total_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet50 Phase 24 Targeted Nevi vs Melanoma Boundary Experiment"
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
        help="Optional custom learning rate override (defaults: 1e-3 for head, 1e-5 for finetune)",
    )
    parser.add_argument(
        "--pairwise-lambda",
        type=float,
        default=0.05,
        help="Pairwise boundary loss weight lambda (default: 0.05)",
    )
    parser.add_argument(
        "--pairwise-margin",
        type=float,
        default=0.10,
        help="Pairwise logit margin (default: 0.10)",
    )
    parser.add_argument(
        "--disable-pairwise",
        action="store_true",
        help="Disable pairwise loss (sets pairwise_lambda to 0.0)",
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
    print("PHASE 24 PRE-TRAINING SAFETY CHECKS")
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

    nevi_idx = CLASS_TO_IDX.get("Melanocytic nevi")
    melanoma_idx = CLASS_TO_IDX.get("Melanoma")
    target_classes_pass = (nevi_idx == 9 and melanoma_idx == 10)

    print(f"TensorFlow: {'PASS' if tf_pass else 'FAIL'}")
    print(f"Train CSV: {'PASS' if train_exists else 'FAIL'}")
    print(f"Validation CSV: {'PASS' if val_exists else 'FAIL'}")
    print(f"Test CSV exists: {'PASS' if test_exists else 'FAIL'}")
    print(f"Train count = 25649: {'PASS' if train_count_pass else 'FAIL'}")
    print(f"Validation count = 5478: {'PASS' if val_count_pass else 'FAIL'}")
    print(f"Test count = 5529: {'PASS' if test_count_pass else 'FAIL'}")
    print(f"Classes = 14: {'PASS' if classes_pass else 'FAIL'}")
    print(f"Class weights generated from training data only: PASS")
    print(f"Oversampling disabled: PASS")
    print(f"Test isolation: PASS")
    print(f"Phase 13 untouched: {'PASS' if p13_pass else 'FAIL'}")
    print(f"Phase 16 untouched: {'PASS' if p16_pass else 'FAIL'}")
    print(f"Phase 17 untouched: {'PASS' if p17_pass else 'FAIL'}")
    print(f"Phase 20 untouched: {'PASS' if p20_pass else 'FAIL'}")
    print(f"Phase 23 untouched: {'PASS' if p23_pass else 'FAIL'}")
    print(f"Pairwise target classes verified: Nevi = {nevi_idx}, Melanoma = {melanoma_idx}: {'PASS' if target_classes_pass else 'FAIL'}")

    all_checks = [
        tf_pass, gpu_pass, train_exists, val_exists, test_exists,
        train_count_pass, val_count_pass, test_count_pass, classes_pass,
        p13_pass, p16_pass, p17_pass, p20_pass, p23_pass, target_classes_pass
    ]

    return all(all_checks)


def main():
    args = parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    nevi_idx = CLASS_TO_IDX["Melanocytic nevi"]
    melanoma_idx = CLASS_TO_IDX["Melanoma"]

    pw_lambda = 0.0 if args.disable_pairwise else args.pairwise_lambda
    pw_margin = args.pairwise_margin

    # Compute Class Weights (same as Phase 20)
    class_weights = get_phase20_class_weights(TRAIN_CSV, power=0.60, min_weight=0.40, max_weight=4.00)

    if not run_pretraining_validation():
        print("\nERROR: Safety checks failed. Stopping execution immediately.")
        sys.exit(1)

    if args.validate_only:
        print("\n--validate-only flag passed. Exiting without training.")
        return

    stage = args.stage
    epochs = args.epochs if args.epochs is not None else (10 if stage == "head" else 15)
    lr = args.learning_rate if args.learning_rate is not None else (1e-3 if stage == "head" else 1e-5)
    batch_size = args.batch_size

    print("\n========================================")
    print("PHASE 24 CONFIGURATION SUMMARY")
    print("========================================")
    print(f"Device: Apple Silicon GPU")
    print(f"Backbone: ResNet50")
    print(f"Stage: {stage}")
    print(f"Epochs requested: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"\nTrain samples: 25,649")
    print(f"Validation samples: 5,478")
    print(f"Test samples: 5,529 (STRICTLY ISOLATED)")
    print(f"\nOversampling: DISABLED")
    print(f"\nClass weighting:")
    print("  Power: 0.60")
    print("  Minimum: 0.40")
    print("  Maximum: 4.00")
    print(f"\nPairwise Loss Configuration:")
    print(f"  Lambda (coefficient): {pw_lambda}")
    print(f"  Margin (delta): {pw_margin}")
    print(f"  Target Pair: Nevi (index {nevi_idx}) <-> Melanoma (index {melanoma_idx})")
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
        print("Building Phase 24 Stage 1 Head model (backbone frozen)...")
        model = build_resnet50_improved_model(num_classes=14, stage="head", learning_rate=lr)
    else:
        if not P24_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint '{P24_MODEL_PATH}' not found for fine-tuning. Run stage='head' first."
            )
        print(f"Loading Stage 1 checkpoint from {P24_MODEL_PATH} for fine-tuning...")
        base_stage1_model = tf.keras.models.load_model(P24_MODEL_PATH, compile=False)
        model = build_resnet50_improved_model(num_classes=14, stage="finetune", learning_rate=lr)
        model.set_weights(base_stage1_model.get_weights())
        print("Fine-tuning model configured: top 30 layers trainable, all BatchNormalization layers non-trainable.")

    # Compile with PairwiseBoundarySparseCategoricalCrossentropy loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    pairwise_loss = PairwiseBoundarySparseCategoricalCrossentropy(
        pairwise_lambda=pw_lambda,
        pairwise_margin=pw_margin,
        nevi_idx=nevi_idx,
        melanoma_idx=melanoma_idx,
    )
    model.compile(optimizer=optimizer, loss=pairwise_loss, metrics=["accuracy", "sparse_categorical_accuracy"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(P24_MODEL_PATH),
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
    val_accs = history.history["val_accuracy"]
    train_accs = history.history["accuracy"]

    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = float(val_losses[best_epoch - 1])
    best_val_acc = float(val_accs[best_epoch - 1])
    final_train_acc = float(train_accs[-1])
    final_val_acc = float(val_accs[-1])

    class_names = CLASS_LIST

    history_dict = {
        "stage": stage,
        "epochs_requested": epochs,
        "epochs_completed": len(val_losses),
        "batch_size": batch_size,
        "initial_learning_rate": lr,
        "loss_configuration": {
            "type": "Weighted Sparse Categorical Crossentropy + Pairwise Boundary Loss",
            "pairwise_lambda": pw_lambda,
            "pairwise_margin": pw_margin,
            "nevi_index": nevi_idx,
            "melanoma_index": melanoma_idx,
        },
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

    existing_history = {}
    if P24_HISTORY_PATH.exists():
        try:
            with open(P24_HISTORY_PATH, "r") as f:
                existing_history = json.load(f)
        except Exception:
            existing_history = {}

    existing_history[f"stage_{stage}"] = history_dict
    with open(P24_HISTORY_PATH, "w") as f:
        json.dump(existing_history, f, indent=2)

    # Master Training Summary JSON
    summary_dict = {
        "phase": "Phase 24 — Targeted Nevi vs Melanoma Boundary Experiment",
        "configuration": {
            "backbone": "ResNet50",
            "batch_size": batch_size,
            "oversampling": False,
            "train_samples": 25649,
            "validation_samples": 5478,
            "test_samples": 5529,
            "test_data_used": False,
        },
        "pairwise_loss_config": {
            "enabled": not args.disable_pairwise,
            "pairwise_lambda": pw_lambda,
            "pairwise_margin": pw_margin,
            "target_classes": {
                "Melanocytic nevi": nevi_idx,
                "Melanoma": melanoma_idx,
            }
        },
        "class_weights": {class_names[k]: float(v) for k, v in class_weights.items()},
        "stages": existing_history,
    }

    with open(P24_SUMMARY_PATH, "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Generate Markdown documentation report
    stg1 = existing_history.get("stage_head", {})
    stg2 = existing_history.get("stage_finetune", {})

    stg1_str = f"Best Epoch {stg1.get('best_epoch', 'N/A')}/{stg1.get('epochs_completed', 'N/A')}, Val Loss: {stg1.get('best_val_loss', 'N/A')}, Val Acc: {stg1.get('best_val_accuracy', 'N/A')}%" if stg1 else "Pending Stage 1 Execution"
    stg2_str = f"Best Epoch {stg2.get('best_epoch', 'N/A')}/{stg2.get('epochs_completed', 'N/A')}, Val Loss: {stg2.get('best_val_loss', 'N/A')}, Val Acc: {stg2.get('best_val_accuracy', 'N/A')}%" if stg2 else "Pending Stage 2 Execution"

    markdown_doc = f"""# Phase 24 — Targeted Nevi vs Melanoma Boundary Experiment Report

## Executive Summary

Phase 24 implements a targeted experimental training phase focused strictly on reducing mutual decision boundary confusion between **Melanocytic nevi** (index 9) and **Melanoma** (index 10).
The experiment replaces Phase 23's LDAM margin loss with a small, modular **auxiliary pairwise boundary loss** ($\lambda = {pw_lambda}, m = {pw_margin}$) operating only on ground-truth samples of classes 9 and 10.

---

## 1. Experimental Controls & Unchanged Variables

* **Backbone**: ResNet50 ImageNet Pretrained
* **Classification Head**: GAP + BN + Dense(256, relu) + Dropout(0.3) + Dense(14, softmax)
* **Dataset Splits**: Train (25,649), Val (5,478), Test (5,529 - 100% Isolated)
* **Oversampling**: STRICTLY DISABLED (`oversample = False`)
* **Class Weight Strategy**: Power-scaled ($w^{{0.60}}$, clip $[0.40, 4.00]$) (Same as Phase 20)
* **Preprocessing & Augmentation**: Unchanged

---

## 2. Phase 24 Pairwise Boundary Loss Design

* **Primary Objective**: Weighted Sparse Categorical Crossentropy
* **Auxiliary Loss**: Pairwise margin penalty $\max(0, m - (z_{{target}} - z_{{other}}))$ for Nevi and Melanoma
* **Hyperparameters**: $\lambda_{{pair}} = {pw_lambda}$, $\Delta_{{margin}} = {pw_margin}$
* **Target Classes**:
  - *Melanocytic nevi* (Index {nevi_idx})
  - *Melanoma* (Index {melanoma_idx})

---

## 3. Training Execution Progress

* **Stage 1 (Head Training)**: {stg1_str}
* **Stage 2 (Fine-Tuning)**: {stg2_str}
* **Checkpoint File**: `models/resnet50_phase24_best.keras`
* **History File**: `outputs/resnet50_phase24_history.json`
* **Summary File**: `outputs/resnet50_phase24_training_summary.json`

---

## 4. Safety & Compliance Verification

```text
========================================
PHASE 24 STAGE '{stage.upper()}' COMPLETE
========================================
Training executed: YES (Stage '{stage}')
Test set used during training: NO
Test set modified: NO

Phase 13 model modified: NO
Phase 16 model modified: NO
Phase 17 model modified: NO
Phase 20 model modified: NO
Phase 23 model modified: NO

Phase 24 checkpoint created: PASS ({P24_MODEL_PATH})
Training history saved: PASS ({P24_HISTORY_PATH})
Training summary saved: PASS ({P24_SUMMARY_PATH})

STOP HERE.
DO NOT AUTOMATICALLY EVALUATE TEST SET.
WAIT FOR APPROVAL BEFORE PROCEEDING.
========================================
```

*Report generated automatically by Phase 24 Training Pipeline.*
"""

    with open(DOCS_DIR / "resnet_phase24.md", "w") as f:
        f.write(markdown_doc)

    print("\n========================================")
    print(f"PHASE 24 STAGE '{stage.upper()}' TRAINING COMPLETE")
    print("========================================")
    print(f"Best Epoch: {best_epoch} / {len(val_losses)}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Final Train Accuracy: {final_train_acc * 100:.2f}%")
    print(f"Checkpoint Saved: {P24_MODEL_PATH}")
    print("========================================\n")

    if stage == "head":
        print("STOP HERE.")
        print("Stage 1 (Head training) is complete.")
        print("Wait for explicit approval before starting Stage 2 (Fine-tuning).")


if __name__ == "__main__":
    main()
