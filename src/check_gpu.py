"""Minimal GPU Diagnostic Utility for TensorFlow on macOS."""

import tensorflow as tf


def check_gpu() -> None:
    """Detect and print TensorFlow hardware devices and GPU status."""
    tf_version = tf.__version__
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    gpu_available = len(gpus) > 0

    print("=" * 50)
    print("TENSORFLOW DEVICE CHECK")
    print("=" * 50)
    print(f"TensorFlow Version: {tf_version}")
    print("\nAvailable GPU Devices:")

    if gpu_available:
        for gpu in gpus:
            print(f"  - {gpu}")
        print("\nGPU Status:\n  AVAILABLE")
        print("\nTraining Device:\n  GPU")
    else:
        print("  - None")
        print("\nCPU Devices:")
        for cpu in cpus:
            print(f"  - {cpu}")
        print("\nGPU Status:\n  NOT AVAILABLE")
        print("\nTraining Device:\n  CPU")

    print("=" * 50)


if __name__ == "__main__":
    check_gpu()
