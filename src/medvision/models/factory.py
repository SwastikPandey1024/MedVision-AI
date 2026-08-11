"""Extensible model factory for architecture selection (Custom CNN, DenseNet121, EfficientNetB0)."""

from typing import Tuple, Dict, Any
import keras


def build_model(
    architecture: str = "densenet121",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    config: Dict[str, Any] | None = None,
) -> keras.Model:
    """Build classification model based on selected architecture.

    Args:
        architecture: Architecture identifier ('custom_cnn', 'densenet121', 'efficientnetb0').
        input_shape: Image input shape tuple (height, width, channels).
        num_classes: Output class count (1 for binary classification).
        config: Optional full configuration dictionary.

    Returns:
        Uncompiled/Compiled Keras Model instance.
    """
    valid_archs = ("custom_cnn", "densenet121", "efficientnetb0")
    if architecture not in valid_archs:
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {valid_archs}")

    raise NotImplementedError(
        f"Model builder for '{architecture}' will be fully implemented in Phase 3/4."
    )
