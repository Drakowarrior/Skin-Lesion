"""
ResNet50 Data Pipeline for 14-Class Skin Condition Dataset (Phase 17)

Includes:
1. Deterministic 14-class mapping definition.
2. Phase 17 targeted class weight computation (smoothed power-scaling formula).
3. Class-aware minority oversampling for training data ONLY (capped at 3.0x multiplier).
4. Medical-image tailored data augmentation pipeline.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf


CLASS_LIST: List[str] = [
    "Actinic keratoses",             # 0
    "Basal cell carcinoma",          # 1
    "Benign keratosis-like lesions", # 2
    "Chickenpox",                    # 3
    "Cowpox",                        # 4
    "Dermatofibroma",                # 5
    "HFMD",                          # 6
    "Healthy",                       # 7
    "Measles",                       # 8
    "Melanocytic nevi",              # 9
    "Melanoma",                      # 10
    "Monkeypox",                     # 11
    "Squamous cell carcinoma",       # 12
    "Vascular lesions"               # 13
]

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_LIST)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_LIST)}


def get_class_mapping() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    return CLASS_LIST, CLASS_TO_IDX, IDX_TO_CLASS


def get_phase17_class_weights(
    train_csv_path: Path = Path("data/processed/train.csv"),
    smoothing: float = 20.0,
    power: float = 0.75,
    min_weight: float = 0.25,
    max_weight: float = 7.0,
) -> Dict[int, float]:
    """
    Compute Phase 17 targeted class weights using train.csv exclusively.

    Formula:
        effective_count = count + smoothing (20)
        raw_weight = (median_class_count / effective_count) ^ 0.75
        weight = clip(raw_weight, min_weight, max_weight)
    """
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training split '{train_csv_path}' not found.")

    df = pd.read_csv(train_csv_path)
    train_counts = df["label"].value_counts().to_dict()
    for cls in CLASS_LIST:
        train_counts.setdefault(cls, 0)

    counts_arr = np.array([train_counts[cls] for cls in CLASS_LIST], dtype=np.float32)
    median_count = np.median(counts_arr)

    effective_counts = counts_arr + smoothing
    raw_weights = np.power(median_count / effective_counts, power)
    clipped_weights = np.clip(raw_weights, min_weight, max_weight)

    return {idx: float(w) for idx, w in enumerate(clipped_weights)}


def build_data_augmentation() -> tf.keras.Sequential:
    """Construct conservative, medical-image friendly Keras data augmentation layers."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.08),
    ], name="data_augmentation_phase17")


def _load_and_preprocess_image(image_path: tf.Tensor) -> tf.Tensor:
    """Load raw JPEG file, decode to RGB, resize to 224x224, and convert to float32."""
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    return img


def create_phase17_dataset(
    csv_path: Path,
    batch_size: int = 32,
    is_training: bool = False,
    oversample: bool = False,
    max_multiplier: float = 3.0,
    target_count: int = 1000,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, int]:
    """
    Create a tf.data.Dataset pipeline for Phase 17.

    If is_training and oversample are True, performs class-aware oversampling on minority
    classes (e.g. Dermatofibroma, Vascular lesions, Squamous cell carcinoma, Actinic keratoses),
    capping oversampling to max_multiplier (3.0x) or target_count (1,000 samples).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)

    if is_training and oversample:
        oversampled_dfs = []
        for cls_name, group_df in df.groupby("label"):
            count = len(group_df)
            if count < target_count:
                desired_count = min(int(count * max_multiplier), target_count)
                # Resample with replacement if desired > count
                resampled_group = group_df.sample(n=desired_count, replace=True, random_state=seed)
                oversampled_dfs.append(resampled_group)
            else:
                oversampled_dfs.append(group_df)
        df = pd.concat(oversampled_dfs, ignore_index=True)

    num_samples = len(df)
    image_paths = df["image_path"].values
    labels = df["label"].map(CLASS_TO_IDX).values

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_samples, seed=seed, reshuffle_each_iteration=True)

    augmentation = build_data_augmentation() if is_training else None

    def process_element(path_tensor, label_tensor):
        img = _load_and_preprocess_image(path_tensor)
        if is_training and augmentation is not None:
            img = augmentation(img, training=True)
        img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img)
        return img_preprocessed, label_tensor

    dataset = dataset.map(process_element, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, num_samples


# Maintain backward compatibility aliases
get_class_weights = get_phase17_class_weights
create_dataset = create_phase17_dataset
