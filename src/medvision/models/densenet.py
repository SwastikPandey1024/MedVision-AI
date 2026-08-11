"""DenseNet121 transfer learning and fine-tuning model builder (Phase 4 & 5)."""

from typing import Tuple
import keras
from keras import layers


def build_densenet121(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: str = "imagenet",
    dense_units: int = 256,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """Build DenseNet121 transfer learning model with custom classification head.

    Args:
        input_shape: Image input dimensions.
        weights: Pre-trained weights initialization ('imagenet' or None).
        dense_units: Units in top dense layer.
        dropout_rate: Dropout probability.

    Returns:
        Keras Model object.
    """
    raise NotImplementedError("DenseNet121 transfer learning model will be built in Phase 4.")


def unfreeze_densenet_layers(model: keras.Model, num_unfreeze_layers: int = 20) -> keras.Model:
    """Unfreeze top N layers of DenseNet121 backbone for fine-tuning.

    Args:
        model: DenseNet121 model instance.
        num_unfreeze_layers: Number of top layers to make trainable.

    Returns:
        Updated Keras Model instance.
    """
    raise NotImplementedError("DenseNet121 layer fine-tuning will be implemented in Phase 5.")
