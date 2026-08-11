"""Custom CNN baseline model architecture for MedVision-AI."""

from typing import Tuple
import keras
from keras import layers


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    initial_filters: int = 32,
    conv_blocks: int = 3,
    dropout_rate: float = 0.3,
    num_classes: int = 1,
    name: str = "CustomCNNBaseline",
) -> keras.Model:
    """Build lightweight Custom CNN baseline model.

    Args:
        input_shape: Image input dimensions (height, width, channels).
        initial_filters: Number of filters in the first conv block.
        conv_blocks: Number of stacked convolutional blocks (default 3).
        dropout_rate: Dropout rate for conv blocks (head uses 0.5).
        num_classes: Number of output classes (1 for binary classification).
        name: Model instance name.

    Returns:
        Keras Model object.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = inputs

    filters = initial_filters
    for block_idx in range(conv_blocks):
        block_name = f"block{block_idx + 1}"

        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            name=f"{block_name}_conv1",
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
        x = layers.Activation("relu", name=f"{block_name}_act1")(x)

        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            name=f"{block_name}_conv2",
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
        x = layers.Activation("relu", name=f"{block_name}_act2")(x)

        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{block_name}_pool")(x)
        x = layers.Dropout(dropout_rate, name=f"{block_name}_drop")(x)

        filters *= 2

    # Global Average Pooling & Dense Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.5, name="head_drop")(x)

    # Note: Explicit dtype="float32" ensures numerical stability under mixed_float16
    outputs = layers.Dense(
        num_classes,
        activation="sigmoid" if num_classes == 1 else "softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model
