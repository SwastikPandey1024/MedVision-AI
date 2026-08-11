"""DenseNet121 transfer learning model architecture for MedVision-AI."""

from typing import Tuple
import keras
from keras import layers


def build_densenet121(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: str = "imagenet",
    dense_units: int = 256,
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    num_classes: int = 1,
    name: str = "DenseNet121Primary",
) -> keras.Model:
    """Build DenseNet121 transfer learning model for chest X-ray classification.

    Args:
        input_shape: Input image shape (height, width, channels).
        weights: Pretrained weights identifier ('imagenet' or None).
        dense_units: Number of units in custom dense classification head.
        dropout_rate: Dropout rate for dense head.
        freeze_backbone: If True, freezes all backbone layers initially.
        num_classes: Output class count (1 for binary classification).
        name: Model instance name.

    Returns:
        Keras Model object.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Load DenseNet121 base model
    base_model = keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
        pooling="avg",
    )

    if freeze_backbone:
        base_model.trainable = False

    x = base_model.output

    # Custom classification head for Pneumonia detection
    x = layers.Dense(dense_units, activation="relu", name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout_rate, name="head_drop")(x)

    # Float32 output layer for mixed_float16 numerical stability
    outputs = layers.Dense(
        num_classes,
        activation="sigmoid" if num_classes == 1 else "softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def unfreeze_densenet_for_finetuning(
    model: keras.Model,
    unfreeze_layers: int = 20,
) -> keras.Model:
    """Safely unfreeze top N layers of DenseNet121 backbone while keeping BatchNormalization layers frozen.

    CRITICAL CTO REQUIREMENT:
    All BatchNormalization layers in the backbone MUST remain non-trainable (trainable = False)
    during fine-tuning to prevent destroying ImageNet running mean & variance statistics.

    Args:
        model: DenseNet121 Keras model instance.
        unfreeze_layers: Number of top layers from the end of the backbone to unfreeze.

    Returns:
        Updated Keras Model.
    """
    # Find base model (backbone) layer inside model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or hasattr(layer, "layers"):
            base_model = layer
            break

    if base_model is not None:
        base_model.trainable = True
        total_layers = len(base_model.layers)
        unfreeze_start_idx = max(0, total_layers - unfreeze_layers)

        for idx, layer in enumerate(base_model.layers):
            if idx >= unfreeze_start_idx and not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        total_layers = len(model.layers)
        unfreeze_start_idx = max(0, total_layers - unfreeze_layers)
        for idx, layer in enumerate(model.layers):
            if idx >= unfreeze_start_idx and not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
            else:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False

    return model
