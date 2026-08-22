"""Unit and integration tests for MedVision-AI Grad-CAM explainability engine."""

import pytest
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

from medvision.explainability.gradcam import (
    auto_detect_target_conv_layer,
    compute_gradcam_heatmap,
    overlay_heatmap,
    generate_gradcam_explanation,
)
from medvision.models.baseline_cnn import build_custom_cnn
from medvision.utils.model_loader import load_medvision_model


@pytest.fixture
def dummy_model():
    """Create a lightweight functional CNN for fast testing."""
    inputs = keras.layers.Input(shape=(64, 64, 3), name="test_input")
    x = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="target_conv")(inputs)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense_feat")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="DummyTestCNN")
    return model


def test_auto_detect_target_conv_layer(dummy_model):
    """Test dynamic layer discovery selects the last 4D conv layer."""
    layer_name = auto_detect_target_conv_layer(dummy_model)
    assert layer_name == "target_conv"


def test_compute_gradcam_heatmap_valid_output(dummy_model):
    """Verify heatmap computation produces normalized float32 array in [0, 1]."""
    test_tensor = tf.random.uniform((1, 64, 64, 3), 0.0, 1.0, dtype=tf.float32)
    heatmap = compute_gradcam_heatmap(
        model=dummy_model,
        image_tensor=test_tensor,
        target_layer_name="target_conv",
    )

    assert isinstance(heatmap, np.ndarray)
    assert len(heatmap.shape) == 2
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_overlay_heatmap_rgb_and_dimensions():
    """Test overlay generation superimposes heatmap onto original image cleanly."""
    heatmap = np.random.uniform(0.0, 1.0, size=(16, 16)).astype(np.float32)
    orig_img = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)

    overlay = overlay_heatmap(heatmap=heatmap, original_image=orig_img, alpha=0.4)

    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (128, 128, 3)
    assert overlay.dtype == np.uint8


def test_overlay_heatmap_grayscale_input():
    """Test overlay generation handles 2D grayscale images gracefully."""
    heatmap = np.random.uniform(0.0, 1.0, size=(16, 16)).astype(np.float32)
    gray_img = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)

    overlay = overlay_heatmap(heatmap=heatmap, original_image=gray_img, alpha=0.5)

    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8


def test_invalid_input_tensor_raises(dummy_model):
    """Test invalid input dimensions and NaN inputs raise appropriate errors."""
    # 5D tensor should raise ValueError
    invalid_5d = tf.random.uniform((1, 2, 64, 64, 3))
    with pytest.raises(ValueError):
        compute_gradcam_heatmap(dummy_model, invalid_5d)

    # NaN tensor should raise ValueError
    nan_tensor = np.full((1, 64, 64, 3), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        compute_gradcam_heatmap(dummy_model, nan_tensor)


def test_model_weights_not_mutated_during_gradcam(dummy_model):
    """Verify that computing Grad-CAM does not modify model weights."""
    weights_before = [w.numpy().copy() for w in dummy_model.trainable_weights]

    test_tensor = tf.random.uniform((1, 64, 64, 3), 0.0, 1.0, dtype=tf.float32)
    _ = compute_gradcam_heatmap(dummy_model, test_tensor, target_layer_name="target_conv")

    weights_after = [w.numpy() for w in dummy_model.trainable_weights]

    assert len(weights_before) == len(weights_after)
    for wb, wa in zip(weights_before, weights_after):
        np.testing.assert_array_equal(wb, wa)


def test_generate_gradcam_explanation_pipeline(dummy_model):
    """Test the end-to-end explanation pipeline produces all required keys."""
    test_tensor = np.random.uniform(0.0, 1.0, size=(1, 64, 64, 3)).astype(np.float32)
    orig_pil = Image.fromarray((test_tensor[0] * 255).astype(np.uint8))

    explanation = generate_gradcam_explanation(
        model=dummy_model,
        preprocessed_tensor=test_tensor,
        original_image=orig_pil,
        target_layer_name="target_conv",
        threshold=0.60,
    )

    assert "probability" in explanation
    assert "prediction" in explanation
    assert "is_pneumonia" in explanation
    assert "raw_heatmap" in explanation
    assert "overlay" in explanation
    assert "side_by_side" in explanation
    assert 0.0 <= explanation["probability"] <= 1.0
    assert explanation["prediction"] in ["Pneumonia", "Normal"]
