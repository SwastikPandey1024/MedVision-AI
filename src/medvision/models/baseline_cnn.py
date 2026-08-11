"""Custom CNN baseline model architecture placeholder (Phase 3)."""

from typing import Tuple
import keras
from keras import layers


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    initial_filters: int = 32,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Build lightweight Custom CNN baseline model.

    Args:
        input_shape: Image input dimensions (height, width, channels).
        initial_filters: Number of filters in the first conv block.
        dropout_rate: Dropout probability.

    Returns:
        Keras Model object.
    """
    raise NotImplementedError("Custom CNN baseline model will be built in Phase 3.")
