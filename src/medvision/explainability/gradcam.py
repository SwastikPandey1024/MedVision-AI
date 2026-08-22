"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for MedVision-AI.

Provides visual interpretability for DenseNet121 and convolutional neural networks
by computing gradients of the target class score with respect to the final convolutional
feature maps.
"""

from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def auto_detect_target_conv_layer(model: keras.Model) -> str:
    """Dynamically discover the final suitable 4D feature map layer in a Keras Model.

    Searches backward through the model layers for the last layer producing a 4D output
    tensor (batch, height, width, channels) suitable for Grad-CAM.

    Args:
        model: Trained Keras classification model.

    Returns:
        Name string of the discovered target layer.

    Raises:
        ValueError: If no suitable 4D convolutional layer is found.
    """
    # Priority known candidates for DenseNet architectures
    priority_candidates = [
        "conv5_block16_2_conv",
        "relu",
        "conv5_block16_concat",
        "bn",
    ]
    model_layer_names = {layer.name for layer in model.layers}
    for candidate in priority_candidates:
        if candidate in model_layer_names:
            return candidate

    # Traverse backwards to find the last 4D output layer
    for layer in reversed(model.layers):
        out_shape = getattr(layer, "output_shape", None)
        if out_shape is None and hasattr(layer, "output"):
            out_shape = layer.output.shape

        if out_shape is not None and len(out_shape) == 4:
            return layer.name

    raise ValueError("No 4D convolutional layer found in model architecture.")


def compute_gradcam_heatmap(
    model: keras.Model,
    image_tensor: Union[tf.Tensor, np.ndarray],
    target_layer_name: Optional[str] = None,
    class_index: int = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute normalized 2D Grad-CAM heatmap array for a given input tensor.

    Args:
        model: Trained Keras classification model.
        image_tensor: Preprocessed image batch tensor of shape (1, H, W, C).
        target_layer_name: Name of target convolutional layer. If None, auto-detected.
        class_index: Output index to compute gradients against.
        eps: Small epsilon constant to prevent division by zero.

    Returns:
        2D Float32 numpy array of shape (H_feat, W_feat) normalized to [0.0, 1.0].

    Raises:
        ValueError: If image_tensor has invalid shape or target layer cannot be found.
    """
    # Validate input tensor shape
    if not isinstance(image_tensor, (tf.Tensor, np.ndarray)):
        raise TypeError(f"image_tensor must be a tf.Tensor or np.ndarray, got {type(image_tensor)}")

    tensor_shape = image_tensor.shape
    if len(tensor_shape) == 3:
        image_tensor = tf.expand_dims(image_tensor, axis=0)
    elif len(tensor_shape) != 4 or tensor_shape[0] != 1:
        raise ValueError(
            f"Expected image_tensor of shape (1, H, W, C) or (H, W, C), got {tensor_shape}"
        )

    if not tf.is_tensor(image_tensor):
        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
    else:
        image_tensor = tf.cast(image_tensor, tf.float32)

    # Check for NaN / Inf in input
    if tf.reduce_any(tf.math.is_nan(image_tensor)) or tf.reduce_any(tf.math.is_inf(image_tensor)):
        raise ValueError("Input tensor contains NaN or Inf values.")

    # Resolve target layer
    if target_layer_name is None:
        target_layer_name = auto_detect_target_conv_layer(model)

    try:
        target_layer = model.get_layer(target_layer_name)
    except ValueError as e:
        raise ValueError(f"Target layer '{target_layer_name}' not found in model.") from e

    # Construct gradient extraction model without mutating original model weights
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        # Select target class score
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            loss = predictions[:, class_index]
        else:
            loss = predictions[:, 0]

    # Compute gradients of class score with respect to feature maps
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError(
            f"Gradients could not be computed for layer '{target_layer_name}'. "
            "Ensure the layer is part of the computational graph leading to the output."
        )

    # Global Average Pooling of gradients across spatial dimensions (height, width)
    # Shape: (1, C)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # Weight feature map channels by pooled gradients
    # conv_outputs shape: (1, H, W, C)
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    # Weighted combination of channels
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to keep only features that have a positive influence on the target class
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize heatmap to [0.0, 1.0]
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / (max_val + eps)
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy().astype(np.float32)


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: Union[np.ndarray, Image.Image],
    alpha: float = 0.4,
    colormap: str = "COLORMAP_JET",
) -> np.ndarray:
    """Superimpose Grad-CAM heatmap onto the original image.

    Args:
        heatmap: 2D Float32 numpy array representing normalized heatmap in [0.0, 1.0].
        original_image: Original RGB or Grayscale image array or PIL Image.
        alpha: Heatmap blend opacity in [0.0, 1.0].
        colormap: OpenCV colormap name (e.g., 'COLORMAP_JET', 'COLORMAP_INFERNO').

    Returns:
        RGB uint8 numpy array of shape (H, W, 3) representing superimposed image.
    """
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = np.array(original_image)

    # Normalize original image to uint8 [0, 255] RGB
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    # Convert grayscale to 3-channel RGB if needed
    if len(img_np.shape) == 2:
        img_rgb = np.stack([img_np] * 3, axis=-1)
    elif len(img_np.shape) == 3:
        if img_np.shape[2] == 1:
            img_rgb = np.concatenate([img_np] * 3, axis=-1)
        elif img_np.shape[2] >= 3:
            img_rgb = img_np[:, :, :3]
        else:
            raise ValueError(f"Unexpected image shape: {img_np.shape}")
    else:
        raise ValueError(f"Unexpected image dimensions: {img_np.shape}")

    orig_h, orig_w = img_rgb.shape[:2]

    # Resize heatmap to match original image dimensions exactly
    heatmap_pil = Image.fromarray(np.uint8(255.0 * np.clip(heatmap, 0.0, 1.0)))
    heatmap_resized = np.array(
        heatmap_pil.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
    )

    # Apply colormap
    if HAS_CV2:
        cmap_code = getattr(cv2, colormap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cmap_code)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    else:
        # High-quality fallback colormap if cv2 is not installed
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
        colored = cmap(heatmap_resized / 255.0)[:, :, :3]  # drop alpha
        heatmap_colored = (colored * 255.0).astype(np.uint8)

    # Blend original image with colored heatmap
    alpha = np.clip(alpha, 0.0, 1.0)
    superimposed = (1.0 - alpha) * img_rgb.astype(np.float32) + alpha * heatmap_colored.astype(np.float32)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed


def generate_gradcam_explanation(
    model: keras.Model,
    preprocessed_tensor: Union[tf.Tensor, np.ndarray],
    original_image: Optional[Union[np.ndarray, Image.Image]] = None,
    target_layer_name: Optional[str] = None,
    class_index: int = 0,
    alpha: float = 0.4,
    threshold: float = 0.60,
) -> Dict[str, Any]:
    """Complete end-to-end Grad-CAM inference and visual explanation pipeline.

    Args:
        model: Trained Keras classification model.
        preprocessed_tensor: Preprocessed tensor of shape (1, 224, 224, 3) in [0, 1].
        original_image: Optional original image for display overlay (defaults to preprocessed tensor).
        target_layer_name: Optional target convolutional layer name.
        class_index: Class index (default 0).
        alpha: Overlay blend factor (default 0.4).
        threshold: Frozen decision threshold (default 0.60).

    Returns:
        Dictionary containing prediction probability, binary decision, raw heatmap,
        normalized heatmap, overlay array, and visualization components.
    """
    if target_layer_name is None:
        target_layer_name = auto_detect_target_conv_layer(model)

    if not tf.is_tensor(preprocessed_tensor):
        preprocessed_tensor = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)
    if len(preprocessed_tensor.shape) == 3:
        preprocessed_tensor = tf.expand_dims(preprocessed_tensor, axis=0)

    # Model inference
    preds = model(preprocessed_tensor, training=False)
    pred_prob = float(preds[0, 0].numpy() if tf.is_tensor(preds) else preds[0, 0])
    is_pneumonia = bool(pred_prob >= threshold)

    # Grad-CAM heatmap
    heatmap = compute_gradcam_heatmap(
        model=model,
        image_tensor=preprocessed_tensor,
        target_layer_name=target_layer_name,
        class_index=class_index,
    )

    # Display image fallback
    if original_image is None:
        display_img = np.uint8(np.clip(preprocessed_tensor[0].numpy() * 255.0, 0, 255))
    elif isinstance(original_image, Image.Image):
        display_img = np.array(original_image)
    else:
        display_img = original_image

    # Overlay
    overlay = overlay_heatmap(
        heatmap=heatmap,
        original_image=display_img,
        alpha=alpha,
    )

    # Side-by-side comparison
    orig_rgb = display_img
    if len(orig_rgb.shape) == 2:
        orig_rgb = np.stack([orig_rgb] * 3, axis=-1)
    elif orig_rgb.shape[-1] == 1:
        orig_rgb = np.concatenate([orig_rgb] * 3, axis=-1)

    # Ensure same height for side-by-side
    h1, w1 = orig_rgb.shape[:2]
    h2, w2 = overlay.shape[:2]
    if (h1, w1) != (h2, w2):
        overlay_resized = np.array(Image.fromarray(overlay).resize((w1, h1)))
    else:
        overlay_resized = overlay

    side_by_side = np.concatenate([orig_rgb, overlay_resized], axis=1)

    return {
        "probability": pred_prob,
        "prediction": "Pneumonia" if is_pneumonia else "Normal",
        "is_pneumonia": is_pneumonia,
        "threshold": threshold,
        "target_layer": target_layer_name,
        "raw_heatmap": heatmap,
        "overlay": overlay,
        "side_by_side": side_by_side,
        "original_image": orig_rgb,
    }
