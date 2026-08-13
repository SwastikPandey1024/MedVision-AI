"""Unit tests proving safe resume training mechanics in MedVision-AI."""

import os
import pytest
import numpy as np
import tensorflow as tf
import keras

from medvision.models.trainer import (
    find_valid_resume_checkpoint,
    get_resume_monitor_baseline,
    train_model,
    validate_resume_checkpoint,
    verify_checkpoint_persistence,
)


def create_dummy_dataset():
    """Create a minimal 2-step dummy dataset for fast unit test execution."""
    x = np.random.randn(4, 224, 224, 3).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
    return ds


def build_and_save_dummy_model(ckpt_path: str):
    """Build, train for one epoch, and save a complete Keras checkpoint."""
    inp = keras.layers.Input(shape=(224, 224, 3))
    x = keras.layers.GlobalAveragePooling2D()(inp)
    out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    model.fit(create_dummy_dataset(), epochs=1, steps_per_epoch=2, verbose=0)
    model.save(ckpt_path)
    return model


def test_missing_resume_checkpoint_fails_clearly(tmp_path):
    """Prove missing resume checkpoint path raises a clear FileNotFoundError."""
    missing_ckpt = str(tmp_path / "non_existent_model.keras")
    ds = create_dummy_dataset()

    with pytest.raises(FileNotFoundError, match="RESUME CHECKPOINT FAILURE"):
        train_model(
            train_ds=ds,
            val_ds=ds,
            epochs=2,
            resume_from=missing_ckpt,
        )


def test_validate_resume_checkpoint_restores_optimizer_state(tmp_path):
    """A valid .keras checkpoint must be loadable with optimizer state intact."""
    checkpoint_path = tmp_path / "saved_stage1.keras"
    build_and_save_dummy_model(str(checkpoint_path))

    result = validate_resume_checkpoint(checkpoint_path)

    assert result.path == checkpoint_path.resolve()
    assert result.size_bytes > 0
    assert result.optimizer_iterations == 2
    assert result.model.optimizer is not None


def test_validate_resume_checkpoint_rejects_empty_file(tmp_path):
    """An empty artifact is never eligible for auto-resume."""
    checkpoint_path = tmp_path / "empty.keras"
    checkpoint_path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty or not a file"):
        validate_resume_checkpoint(checkpoint_path)

    assert find_valid_resume_checkpoint(checkpoint_path, "densenet121") is None


def test_resume_uses_best_historical_monitor_value(tmp_path):
    """A resumed best-only checkpoint must retain the prior best PR-AUC threshold."""
    csv_path = tmp_path / "history.csv"
    csv_path.write_text("epoch,val_pr_auc\n0,0.561\n1,0.549\n", encoding="utf-8")

    baseline = get_resume_monitor_baseline(
        model=None,
        val_ds=None,
        validation_steps=None,
        csv_log_path=str(csv_path),
    )

    assert baseline == pytest.approx(0.561)


def test_existing_checkpoint_recovers_epoch_from_optimizer_state(tmp_path, capsys):
    """Prove a complete checkpoint resumes after its completed epoch."""
    ckpt_path = str(tmp_path / "saved_stage1.keras")
    build_and_save_dummy_model(ckpt_path)

    out_ckpt = str(tmp_path / "resumed_stage1_best.keras")
    ds = create_dummy_dataset()

    history = train_model(
        train_ds=ds,
        val_ds=ds,
        epochs=2,
        steps_per_epoch=2,
        validation_steps=2,
        checkpoint_filepath=out_ckpt,
        resume_from=ckpt_path,
    )

    captured = capsys.readouterr()
    assert "RESUME CHECKPOINT:" in captured.out
    assert "OPTIMIZER STEPS  : 2" in captured.out
    assert "RESUME EPOCH     : 1" in captured.out
    assert "TARGET EPOCHS    : 2" in captured.out
    assert verify_checkpoint_persistence(out_ckpt) is True
    # Epoch 1 (index 0) was skipped, so history contains epoch index 1 output
    assert len(history.epoch) == 1
    assert history.epoch[0] == 1


def test_resume_rejects_mismatched_explicit_epoch(tmp_path):
    """Prove an explicit epoch cannot override the checkpoint optimizer state."""
    ckpt_path = str(tmp_path / "saved_stage1_default.keras")
    build_and_save_dummy_model(ckpt_path)

    ds = create_dummy_dataset()

    with pytest.raises(ValueError, match="does not match"):
        train_model(
            train_ds=ds,
            val_ds=ds,
            epochs=2,
            steps_per_epoch=2,
            validation_steps=2,
            resume_from=ckpt_path,
            initial_epoch=0,
        )


def test_normal_non_resume_training_unchanged(tmp_path):
    """Prove normal non-resume training creates model from scratch, starting at initial_epoch 0."""
    out_ckpt = str(tmp_path / "fresh_model_best.keras")
    ds = create_dummy_dataset()
    fresh_model = build_and_save_dummy_model(str(tmp_path / "temp_init.keras"))

    history = train_model(
        model=fresh_model,
        train_ds=ds,
        val_ds=ds,
        epochs=1,
        steps_per_epoch=2,
        validation_steps=2,
        checkpoint_filepath=out_ckpt,
        resume_from=None,
    )

    assert len(history.epoch) == 1
    assert history.epoch[0] == 0
    assert verify_checkpoint_persistence(out_ckpt) is True
