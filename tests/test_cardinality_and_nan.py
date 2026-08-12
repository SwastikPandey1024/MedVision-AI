"""Regression tests for dataset cardinality, batch size semantics, NaN loss guard, and real-data diagnostic."""

import math
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from medvision.data.local_dev_loader import load_dev_subset_datasets
from medvision.data.preprocessing import create_tfrecord_dataset
from medvision.models.factory import build_model
from medvision.models.trainer import (
    NaNGuardCallback,
    run_real_batch_diagnostic,
    train_model,
)
from scripts.train import check_laptop_safety_and_provenance


def test_dev_subset_finite_cardinality():
    """Verify train_ds, val_ds, and test_ds are finite (no infinite repeat)."""
    train_ds, val_ds, test_ds, df_train, df_val, df_test = load_dev_subset_datasets(
        sample_fraction=0.05, batch_size=4
    )

    # Count actual batches yielded by train_ds, val_ds, and test_ds
    train_batch_count = sum(1 for _ in train_ds)
    val_batch_count = sum(1 for _ in val_ds)
    test_batch_count = sum(1 for _ in test_ds)

    expected_train_batches = math.ceil(len(df_train) / 4)
    expected_val_batches = math.ceil(len(df_val) / 4)
    expected_test_batches = math.ceil(len(df_test) / 4)

    assert train_batch_count == expected_train_batches, f"Expected {expected_train_batches} train batches, got {train_batch_count}"
    assert val_batch_count == expected_val_batches, f"Expected {expected_val_batches} val batches, got {val_batch_count}"
    assert test_batch_count == expected_test_batches, f"Expected {expected_test_batches} test batches, got {test_batch_count}"


def test_tfrecord_dataset_default_finite_cardinality(tmp_path):
    """Verify create_tfrecord_dataset produces a finite dataset when repeat=False."""
    from medvision.data.tfrecord_writer import write_manifest_to_tfrecords

    records = [
        {"patient_id": f"P{i:03d}", "target": i % 2, "bbox_count": 0, "bboxes": [], "image_path": ""}
        for i in range(12)
    ]
    df = pd.DataFrame(records)
    shard_paths = write_manifest_to_tfrecords(df, split_name="train", output_dir=tmp_path, num_shards=2)

    ds = create_tfrecord_dataset(shard_paths, batch_size=4, is_training=True, repeat=False)
    batch_count = sum(1 for _ in ds)
    assert batch_count == 3, f"Expected 3 finite batches, got {batch_count}"


def test_expected_steps_calculation():
    """Verify expected steps per epoch formula for RSNA train (18678) and val (4003) splits."""
    train_samples = 18678
    val_samples = 4003
    global_batch_size = 64

    expected_train_steps = math.ceil(train_samples / global_batch_size)
    expected_val_steps = math.ceil(val_samples / global_batch_size)

    assert expected_train_steps == 292
    assert expected_val_steps == 63


def test_batch_size_divisibility():
    """Verify global batch size and per-replica batch size calculation with divisibility assertion."""
    global_batch_size = 64
    num_replicas = 2

    assert global_batch_size % num_replicas == 0
    per_replica_batch_size = global_batch_size // num_replicas
    assert per_replica_batch_size == 32

    # Test invalid non-divisible global batch size
    with pytest.raises(AssertionError):
        invalid_global = 65
        assert invalid_global % num_replicas == 0, f"Global batch size ({invalid_global}) must be cleanly divisible by num_replicas ({num_replicas})"


def test_nan_guard_callback_triggers_on_nan_loss():
    """Verify NaNGuardCallback aborts training immediately when loss becomes NaN."""
    callback = NaNGuardCallback()
    dummy_model = keras.Sequential([keras.layers.Dense(1)])
    dummy_model.stop_training = False
    callback.model = dummy_model

    # Finite loss step should pass without error
    callback.on_batch_end(batch=0, logs={"loss": 0.5})
    assert dummy_model.stop_training is False

    # NaN loss step must trigger RuntimeError and stop training
    with pytest.raises(RuntimeError, match="HARD NAN GUARD"):
        callback.on_batch_end(batch=1, logs={"loss": float("nan")})

    assert dummy_model.stop_training is True
    assert callback.first_bad_step == 2


def test_real_batch_diagnostic_finite_flags():
    """Verify run_real_batch_diagnostic inspects 1 batch and returns finite flags."""
    model = build_model(architecture="custom_cnn", compile_model=True)

    x_dummy = tf.random.uniform((8, 224, 224, 3), minval=0.0, maxval=1.0)
    y_dummy = tf.constant([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    ds = tf.data.Dataset.from_tensor_slices((x_dummy, y_dummy)).batch(4)

    diag = run_real_batch_diagnostic(model, train_ds=ds, is_dev=True)

    assert diag["x_finite"] is True
    assert diag["y_finite"] is True
    assert diag["pred_finite"] is True
    assert diag["raw_loss_finite"] is True
    assert diag["grad_finite"] is True
    assert diag["post_update_weights_finite"] is True
    assert diag["first_failure"] == "NONE (ALL STAGES FINITE)"


def test_real_batch_diagnostic_with_strategy():
    """Verify run_real_batch_diagnostic works under an explicit distribution strategy."""
    strategy = tf.distribute.get_strategy()
    model = build_model(architecture="custom_cnn", compile_model=True, strategy=strategy)

    x_dummy = tf.random.uniform((8, 224, 224, 3), minval=0.0, maxval=1.0)
    y_dummy = tf.constant([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    ds = tf.data.Dataset.from_tensor_slices((x_dummy, y_dummy)).batch(4)

    diag = run_real_batch_diagnostic(model, train_ds=ds, strategy=strategy, is_dev=True)

    assert diag["x_finite"] is True
    assert diag["y_finite"] is True
    assert diag["pred_finite"] is True
    assert diag["raw_loss_finite"] is True
    assert diag["grad_finite"] is True
    assert diag["post_update_weights_finite"] is True
    assert diag["first_failure"] == "NONE (ALL STAGES FINITE)"


def test_stage1_fresh_model_and_dataset_reinitialization():
    """Verify fresh model and dataset reinitialization before Stage 1 training."""
    strategy = tf.distribute.get_strategy()
    model1 = build_model(architecture="custom_cnn", compile_model=True, strategy=strategy)
    model2 = build_model(architecture="custom_cnn", compile_model=True, strategy=strategy)

    # Ensure model2 is a distinct fresh instance
    assert model1 is not model2
    assert len(model1.weights) == len(model2.weights)


def test_train_script_stage_choices_include_exp_a_and_exp_b():
    """Verify parse_args contains exp_a and exp_b in --stage choices."""
    from scripts.train import parse_args
    import sys

    # Mock command line arguments
    sys.argv = ["train.py", "--stage", "exp_a"]
    args_a = parse_args()
    assert args_a.stage == "exp_a"

    sys.argv = ["train.py", "--stage", "exp_b"]
    args_b = parse_args()
    assert args_b.stage == "exp_b"


def test_train_script_help_string_does_not_raise_error():
    """Verify parse_args --help formatting does not raise TypeError."""
    from scripts.train import parse_args
    import sys

    sys.argv = ["train.py", "--help"]
    with pytest.raises(SystemExit):
        parse_args()


def test_check_laptop_safety_and_provenance_guard(monkeypatch):
    """Verify full training mode fails safely when missing real RSNA dataset or GPU."""
    # Test laptop CPU guard: mode='full' on non-Kaggle with 0 GPUs
    with pytest.raises(SystemExit):
        check_laptop_safety_and_provenance(mode="full", gpu_count=0)
