"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation (Phase 8)."""

from typing import Tuple
import numpy as np
import tensorflow as tf


def auto_detect_target_conv_layer(model: tf.keras.Model) -> str:
    """Dynamically discover the final 4D Conv2D layer in a Keras Model.

    Args:
        model: Trained Keras classification model.

    Returns:
        Name string of the last convolutional layer.
    """
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:  # (batch, H, W, C)
            return layer.name
    raise ValueError("No 4D convolutional layer found in model architecture.")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    target_layer_name: str | None = None,
    class_index: int = 0,
) -> np.ndarray:
    """Compute Grad-CAM heatmap array for a target conv layer.

    Args:
        model: Trained Keras classification model.
        image_tensor: Preprocessed image batch tensor of shape (1, H, W, C).
        target_layer_name: Name of target convolutional layer. If None, auto-detected.
        class_index: Output index to compute gradients against.

    Returns:
        2D Float32 numpy array representing normalized heatmap [0, 1].
    """
    if target_layer_name is None:
        target_layer_name = auto_detect_target_conv_layer(model)

    raise NotImplementedError("Grad-CAM engine will be implemented in Phase 8.")


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "COLORMAP_JET",
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original input image.

    Args:
        heatmap: 2D heatmap numpy array.
        original_image: RGB/Grayscale image array.
        alpha: Heatmap blend opacity.
        colormap: OpenCV colormap constant string.

    Returns:
        Superimposed RGB image array.
    """
    raise NotImplementedError("Grad-CAM overlay utility will be implemented in Phase 8.")
