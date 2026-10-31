"""
TensorFlow Apple Silicon GPU Diagnostic & Dataset Split Integrity Verification (Phase 10)

This script:
1. Verifies TensorFlow and Keras installation.
2. Detects Apple Silicon GPU acceleration via Metal (PluggableDevice).
3. Performs actual GPU tensor computations.
4. Validates ResNet50 architecture loading with ImageNet weights.
5. Checks for duplicate image content (data leakage) across train, validation, and test splits.
"""

import sys
import os
import hashlib
from pathlib import Path
import tensorflow as tf
import keras


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_gpu():
    print("========================================")
    print("TENSORFLOW APPLE GPU CHECK")
    print("========================================")

    tf_version = tf.__version__
    keras_version = getattr(keras, "__version__", "Available (via tf.keras)")

    print(f"\nTensorFlow Version: {tf_version}")
    print(f"Keras Version: {keras_version}")

    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")

    print("\nCPU Devices:")
    for cpu in cpus:
        print(f"  - {cpu.name}")

    print("\nGPU Devices:")
    if gpus:
        for gpu in gpus:
            print(f"  - {gpu.name}")
        gpu_available = True
    else:
        print("  - None detected")
        gpu_available = False

    print(f"\nApple GPU Available: {'YES' if gpu_available else 'NO'}")

    tensor_pass = False
    compute_pass = False
    selected_device = None

    if gpu_available:
        try:
            # 1. Tensor creation
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            tensor_pass = True

            # 2. GPU execution test
            with tf.device("/device:GPU:0"):
                c = tf.matmul(a, b)
                # Force evaluation
                _ = c.numpy()
                compute_pass = True
                selected_device = "/device:GPU:0 (Apple Metal GPU)"
        except Exception as e:
            print(f"GPU Execution Error: {e}")

    print("\nGPU Test:")
    print(f"Tensor creation: {'PASS' if tensor_pass else 'FAIL'}")
    print(f"GPU computation: {'PASS' if compute_pass else 'FAIL'}")
    if selected_device:
        print(f"\nSelected Device:\n  {selected_device}")

    # ResNet50 Compatibility Test
    print("\n----------------------------------------")
    print("RESNET50 ARCHITECTURE COMPATIBILITY")
    print("----------------------------------------")
    resnet_pass = False
    try:
        model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        dummy_input = tf.random.normal([1, 224, 224, 3])
        output = model(dummy_input)
        if output.shape == (1, 7, 7, 2048):
            resnet_pass = True
            print(f"ResNet50 Loading: PASS")
            print(f"ImageNet Weights: PASS")
            print(f"Input Shape: (1, 224, 224, 3) -> Output Shape: {output.shape}")
        else:
            print(f"ResNet50 Output Shape Mismatch: {output.shape}")
    except Exception as e:
        print(f"ResNet50 Compatibility Test FAIL: {e}")

    # Duplicate Split Check Across Train / Val / Test
    print("\n----------------------------------------")
    print("DUPLICATE SPLIT CHECK")
    print("----------------------------------------")
    
    splits = ["train", "val", "test"]
    split_hashes = {s: set() for s in splits}
    
    raw_data_dir = Path("data/raw")
    for s in splits:
        split_dir = raw_data_dir / s
        if split_dir.exists():
            for root, _, files in os.walk(split_dir):
                for f in files:
                    if f.startswith(".") or not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        continue
                    fp = Path(root) / f
                    h = compute_file_sha256(fp)
                    split_hashes[s].add(h)

    train_val_dup = len(split_hashes["train"].intersection(split_hashes["val"]))
    train_test_dup = len(split_hashes["train"].intersection(split_hashes["test"]))
    val_test_dup = len(split_hashes["val"].intersection(split_hashes["test"]))

    print(f"Train ↔ Validation: {train_val_dup}")
    print(f"Train ↔ Test:       {train_test_dup}")
    print(f"Validation ↔ Test:  {val_test_dup}")

    split_leakage_free = (train_val_dup == 0 and train_test_dup == 0 and val_test_dup == 0)

    print("\n========================================")
    if not gpu_available or not compute_pass:
        print("STATUS: FAIL")
        print("\nTensorFlow GPU acceleration is unavailable.")
        print("Training should not be started.")
        print("========================================")
        return 1

    if not resnet_pass:
        print("STATUS: FAIL")
        print("\nResNet50 model instantiation failed.")
        print("========================================")
        return 1

    if not split_leakage_free:
        print("STATUS: FAIL — DO NOT START TRAINING")
        print("\nCross-split duplicate image content detected.")
        print("Data leakage between splits must be resolved before model training.")
        print("========================================")
        return 1

    print("STATUS: PASS — READY FOR RESNET50 PIPELINE")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(check_gpu())
