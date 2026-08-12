"""Unit tests for Phase 3 model architectures, factory, trainer, and metrics."""

import pytest
import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from medvision.models.baseline_cnn import build_custom_cnn
from medvision.models.densenet import build_densenet121, unfreeze_densenet_for_finetuning
from medvision.models.factory import build_model
from medvision.models.trainer import compute_training_class_weights
from medvision.utils.metrics import get_model_metrics, Specificity, F1Score


def test_custom_cnn_build():
    """Verify Custom CNN shape, parameter counts, and dummy forward pass."""
    model = build_custom_cnn(input_shape=(224, 224, 3), initial_filters=32, num_classes=1)

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)
    assert model.count_params() > 0

    # Dummy forward pass
    dummy_input = np.ones((2, 224, 224, 3), dtype=np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (2, 1)
    assert np.all(output.numpy() >= 0.0) and np.all(output.numpy() <= 1.0)


def test_densenet121_build_and_unfreeze():
    """Verify DenseNet121 construction and safe BatchNorm unfreezing."""
    model = build_densenet121(input_shape=(224, 224, 3), freeze_backbone=True)

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)

    # Test unfreezing
    model = unfreeze_densenet_for_finetuning(model, unfreeze_layers=20)

    # Verify that ALL BatchNormalization layers remain trainable=False
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            assert layer.trainable is False, f"BatchNorm layer '{layer.name}' should remain frozen!"


def test_model_factory():
    """Verify model factory architecture instantiation."""
    cnn_model = build_model(architecture="custom_cnn", compile_model=True)
    assert cnn_model.output_shape == (None, 1)

    dn_model = build_model(architecture="densenet121", compile_model=True)
    assert dn_model.output_shape == (None, 1)

    with pytest.raises(ValueError):
        build_model(architecture="invalid_arch")


def test_strategy_reuse_in_build_model():
    """Verify build_model reuses supplied distribution strategy without re-instantiation."""
    custom_strategy = tf.distribute.get_strategy()

    with custom_strategy.scope():
        model = build_model(
            architecture="custom_cnn",
            compile_model=True,
            strategy=custom_strategy,
        )

    assert model.output_shape == (None, 1)


def test_compute_training_class_weights():
    """Verify class weight calculation on training data only."""
    df_train = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(100)],
        "target": [0] * 80 + [1] * 20,  # 80 normal, 20 pneumonia (20% positive)
    })

    weights = compute_training_class_weights(df_train)

    # W_0 = 100 / (2 * 80) = 0.625
    # W_1 = 100 / (2 * 20) = 2.5
    assert pytest.approx(weights[0], 0.001) == 0.625
    assert pytest.approx(weights[1], 0.001) == 2.500


def test_metrics_instantiation():
    """Verify metric objects initialization and PR-AUC presence."""
    metrics = get_model_metrics()
    metric_names = [m.name for m in metrics]

    assert "pr_auc" in metric_names
    assert "roc_auc" in metric_names
    assert "specificity" in metric_names
    assert "f1_score" in metric_names
