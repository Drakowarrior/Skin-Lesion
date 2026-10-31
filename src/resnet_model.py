The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.
"""
ResNet50 Improved Model Architecture & Transfer Learning Strategy (Phase 15)

Implements:
1. ResNet50 backbone loaded with ImageNet pretrained weights.
2. Refined Classification Head:
   GlobalAveragePooling2D -> BatchNormalization -> Dense(256, relu) -> Dropout(0.3) -> Dense(14, softmax).
3. Stage 1 (head training): Frozen backbone, learning rate 1e-3.
4. Stage 2 (fine-tuning): Unfreezes top ResNet blocks with BatchNormalization layers safely kept frozen, learning rate 1e-5.
"""

from typing import Tuple
import tensorflow as tf


def build_resnet50_improved_model(
    num_classes: int = 14,
    stage: str = "head",
    learning_rate: float = None,
) -> tf.keras.Model:
    """
    Build and compile the improved ResNet50 transfer learning model.

    Args:
        num_classes: Number of output target classes (default 14).
        stage: Training stage - 'head' (frozen backbone) or 'finetune' (partially un-frozen backbone).
        learning_rate: Optional learning rate override. Default is 1e-3 for 'head', 1e-5 for 'finetune'.

    Returns:
        Compiled Keras Model instance.
    """
    # 1. Base Pretrained Backbone
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    if stage == "head":
        # Freeze entire backbone for Stage 1
        base_model.trainable = False
        lr = learning_rate if learning_rate is not None else 1e-3
    elif stage == "finetune":
        # Unfreeze top ResNet blocks (last 30 layers) for Stage 2
        base_model.trainable = True
        # Freeze early layers up to the last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # CRITICAL BATCHNORMALIZATION SAFETY:
        # Keep all BatchNormalization layers non-trainable during fine-tuning to prevent
        # unstable running statistics updates on limited batch sizes.
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        lr = learning_rate if learning_rate is not None else 1e-5
    else:
        raise ValueError(f"Invalid stage '{stage}'. Expected 'head' or 'finetune'.")

    # 2. Refined Classification Head with BatchNormalization and Dropout(0.3)
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = base_model(inputs, training=(stage == "finetune"))
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    head_bn = tf.keras.layers.BatchNormalization(name="head_batch_norm")
    if stage == "finetune":
        head_bn.trainable = False
    x = head_bn(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_0.3")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"ResNet50_SkinLesions_Improved_{stage.capitalize()}"
    )

    if stage == "finetune":
        # Recursively ensure all BatchNormalization layers across model and sub-models are non-trainable
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            elif hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                        sub_layer.trainable = False

    # 3. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy", "sparse_categorical_accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model