"""Unit tests for checkpoint directory creation and post-fit persistence verification."""

import os
import pytest
from pathlib import Path

from medvision.config.settings import get_output_dir, get_output_base_dir
from medvision.models.trainer import verify_checkpoint_persistence


def test_get_output_dir_creates_directories(tmp_path, monkeypatch):
    """Verify get_output_dir automatically creates checkpoints, logs, and metrics subdirectories."""
    test_out = tmp_path / "test_outputs"
    monkeypatch.setenv("MEDVISION_OUTPUT_DIR", str(test_out))

    ckpt_dir = get_output_dir("checkpoints")
    logs_dir = get_output_dir("logs")
    metrics_dir = get_output_dir("metrics")

    assert ckpt_dir.exists() and ckpt_dir.is_dir()
    assert logs_dir.exists() and logs_dir.is_dir()
    assert metrics_dir.exists() and metrics_dir.is_dir()
    assert str(ckpt_dir) == str(test_out / "checkpoints")


def test_stage1_checkpoint_path_format(tmp_path, monkeypatch):
    """Verify Stage 1 checkpoint path resolves to required filename."""
    test_out = tmp_path / "medvision_outputs"
    monkeypatch.setenv("MEDVISION_OUTPUT_DIR", str(test_out))

    stage1_path = get_output_dir("checkpoints") / "densenet121_stage1_best.keras"
    assert stage1_path.name == "densenet121_stage1_best.keras"
    assert stage1_path.parent.name == "checkpoints"


def test_verify_checkpoint_persistence_success(tmp_path, capsys):
    """Verify verify_checkpoint_persistence returns True and prints success message when file exists and > 0 bytes."""
    ckpt_file = tmp_path / "model_best.keras"
    ckpt_file.write_bytes(b"dummy_model_data_bytes_12345")

    result = verify_checkpoint_persistence(str(ckpt_file))
    assert result is True

    captured = capsys.readouterr()
    assert "CHECKPOINT PERSISTENCE: PASS" in captured.out
    assert "CHECKPOINT PATH:" in captured.out
    assert "CHECKPOINT SIZE MB:" in captured.out


def test_verify_checkpoint_persistence_missing_raises(tmp_path):
    """Verify verify_checkpoint_persistence raises RuntimeError if file is missing."""
    missing_file = tmp_path / "non_existent_model.keras"

    with pytest.raises(RuntimeError, match="CHECKPOINT PERSISTENCE FAILURE.*does not exist"):
        verify_checkpoint_persistence(str(missing_file))


def test_verify_checkpoint_persistence_zero_bytes_raises(tmp_path):
    """Verify verify_checkpoint_persistence raises RuntimeError if file has 0 bytes."""
    empty_file = tmp_path / "empty_model.keras"
    empty_file.write_bytes(b"")

    with pytest.raises(RuntimeError, match="CHECKPOINT PERSISTENCE FAILURE.*0 bytes"):
        verify_checkpoint_persistence(str(empty_file))
