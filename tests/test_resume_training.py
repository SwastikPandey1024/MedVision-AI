"""Unit tests proving safe resume training mechanics in MedVision-AI."""

import os
import pytest
import numpy as np
import tensorflow as tf
import keras

from medvision.models.trainer import train_model, build_callbacks, verify_checkpoint_persistence


def create_dummy_dataset():
    """Create a minimal 2-step dummy dataset for fast unit test execution."""
    x = np.random.randn(4, 224, 224, 3).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
    return ds


def build_and_save_dummy_model(ckpt_path: str):
    """Build, compile, and save a minimal Keras model checkpoint."""
    inp = keras.layers.Input(shape=(224, 224, 3))
    out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(inp)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
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


def test_existing_checkpoint_loaded_and_initial_epoch_set(tmp_path, capsys):
    """Prove existing checkpoint is loaded, initial_epoch is accepted, and Epoch 1 is skipped."""
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
        initial_epoch=1,
    )

    captured = capsys.readouterr()
    assert "RESUME CHECKPOINT:" in captured.out
    assert "RESUME EPOCH     : 1" in captured.out
    assert "TARGET EPOCHS    : 2" in captured.out
    assert verify_checkpoint_persistence(out_ckpt) is True
    # Epoch 1 (index 0) was skipped, so history contains epoch index 1 output
    assert len(history.epoch) == 1
    assert history.epoch[0] == 1


def test_resume_default_initial_epoch(tmp_path, capsys):
    """Prove resume_from defaults initial_epoch to 1 when initial_epoch is 0."""
    ckpt_path = str(tmp_path / "saved_stage1_default.keras")
    build_and_save_dummy_model(ckpt_path)

    out_ckpt = str(tmp_path / "resumed_stage1_default_best.keras")
    ds = create_dummy_dataset()

    history = train_model(
        train_ds=ds,
        val_ds=ds,
        epochs=2,
        steps_per_epoch=2,
        validation_steps=2,
        checkpoint_filepath=out_ckpt,
        resume_from=ckpt_path,
        initial_epoch=0,
    )

    captured = capsys.readouterr()
    assert "RESUME EPOCH     : 1" in captured.out
    assert len(history.epoch) == 1
    assert history.epoch[0] == 1


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
        initial_epoch=0,
    )

    assert len(history.epoch) == 1
    assert history.epoch[0] == 0
    assert verify_checkpoint_persistence(out_ckpt) is True
