"""
ResNet50 Phase 27 — Production Model Freeze & Integrity Verification Script

Executes:
1. Verifies existing Phase 26 checkpoint (models/resnet50_phase26_best.keras).
2. Copies frozen model into models/production/resnet50_skin_lesion_v1.keras.
3. Generates SHA-256 hash artifact (models/production/model_sha256.txt).
4. Generates production metadata file (models/production/model_metadata.json).
5. Performs non-test dataset smoke test using SkinLesionPredictor API.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from src.class_labels import CLASS_NAMES, NUM_CLASSES
from src.inference import SkinLesionPredictor


PHASE26_MODEL_PATH = Path("models/resnet50_phase26_best.keras")
PRODUCTION_DIR = Path("models/production")
PRODUCTION_MODEL_PATH = PRODUCTION_DIR / "resnet50_skin_lesion_v1.keras"
SHA256_PATH = PRODUCTION_DIR / "model_sha256.txt"
METADATA_PATH = PRODUCTION_DIR / "model_metadata.json"
TRAIN_CSV = Path("data/processed/train.csv")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("========================================")
    print("PHASE 27 — PRODUCTION MODEL FREEZE")
    print("========================================")

    # 1. Verify Phase 26 Checkpoint Exists
    if not PHASE26_MODEL_PATH.exists():
        print(f"ERROR: Frozen checkpoint not found at '{PHASE26_MODEL_PATH}'. Stopping.")
        sys.exit(1)

    print(f"Checkpoint Verified: {PHASE26_MODEL_PATH}")

    # 2. Load Model in Read-Only Mode (compile=False)
    print("Loading model in read-only mode (compile=False)...")
    model = tf.keras.models.load_model(PHASE26_MODEL_PATH, compile=False)

    input_shape = model.input_shape  # e.g., (None, 224, 224, 3)
    output_shape = model.output_shape  # e.g., (None, 14)
    total_params = model.count_params()
    trainable_params = sum(
        int(tf.keras.backend.count_params(w)) for w in model.trainable_weights
    )

    num_classes = output_shape[-1]
    input_size = [int(dim) for dim in input_shape[1:4]]

    print("\n--- Model Architecture & Specs ---")
    print(f"Input Shape: {input_shape}")
    print(f"Output Shape: {output_shape}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Number of Classes: {num_classes}")
    print(f"Class Names: {CLASS_NAMES}")

    if num_classes != NUM_CLASSES:
        print(f"ERROR: Expected {NUM_CLASSES} classes, found {num_classes}. Stopping.")
        sys.exit(1)

    # 3. Create Production Directory & Copy Checkpoint
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying frozen checkpoint to production directory...")
    shutil.copy2(PHASE26_MODEL_PATH, PRODUCTION_MODEL_PATH)
    print(f"Production Model Created: {PRODUCTION_MODEL_PATH}")

    # 4. Generate SHA-256 Checksum
    print("Computing SHA-256 checksum...")
    sha256_hash = compute_sha256(PRODUCTION_MODEL_PATH)
    with open(SHA256_PATH, "w") as f:
        f.write(f"{sha256_hash}  {PRODUCTION_MODEL_PATH.name}\n")
    print(f"SHA-256 Hash: {sha256_hash}")
    print(f"SHA-256 Saved: {SHA256_PATH}")

    # 5. Generate Model Metadata JSON
    metadata = {
        "model_name": "ResNet50 Skin Lesion Classifier",
        "version": "v1",
        "source_checkpoint": PHASE26_MODEL_PATH.name,
        "production_checkpoint": PRODUCTION_MODEL_PATH.name,
        "architecture": "ResNet50",
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "dataset": "HAM10000 / ISIC Expanded",
        "input_size": input_size,
        "framework": "TensorFlow/Keras",
        "phase": 26,
        "status": "frozen_production_candidate",
        "sha256": sha256_hash,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata Saved: {METADATA_PATH}")

    # 6. Run Production Smoke Test (NON-TEST sample)
    print("\n--- Running Production Inference Smoke Test ---")
    predictor = SkinLesionPredictor(PRODUCTION_MODEL_PATH)

    train_df = pd.read_csv(TRAIN_CSV)
    smoke_sample_path = train_df.iloc[0]["image_path"]
    print(f"Smoke Test Image (Training Split): {smoke_sample_path}")

    res = predictor.predict(smoke_sample_path)
    probs_sum = res["probability_sum"]

    print(f"Predicted Class: {res['predicted_class']} (Index {res['predicted_index']})")
    print(f"Model Confidence: {res['confidence']:.4f}")
    print(f"Probability Sum: {probs_sum:.6f}")
    print("Top 3 Predictions:")
    for t in res["top3"]:
        print(f"  - {t['class']}: {t['confidence']:.4f}")

    # Validation assertions
    valid_shape = (num_classes == 14)
    valid_finite = np.isfinite(probs_sum)
    valid_sum = abs(probs_sum - 1.0) < 1e-4

    if valid_shape and valid_finite and valid_sum:
        print("\nProduction Smoke Test: PASS")
    else:
        print("\nProduction Smoke Test: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
