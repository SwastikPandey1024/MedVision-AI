"""Unit and smoke tests for Streamlit Community Cloud deployment readiness."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from app.services.inference_service import (
    download_model_checkpoint,
    get_configured_model_url,
    resolve_model_checkpoint,
    process_uploaded_image,
)
from app.components.disclaimer import render_disclaimer
from app.components.header import render_header
from app.components.metrics_card import render_performance_summary


def test_streamlit_config_toml_exists():
    """Confirm .streamlit/config.toml exists and contains server and theme configuration."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / ".streamlit" / "config.toml"
    assert config_path.exists(), ".streamlit/config.toml is missing"
    content = config_path.read_text(encoding="utf-8")
    assert "[server]" in content
    assert "[theme]" in content
    assert "headless = true" in content


def test_requirements_contains_all_streamlit_dependencies():
    """Confirm requirements.txt includes all runtime dependencies including pydicom and streamlit."""
    root = Path(__file__).resolve().parent.parent
    req_path = root / "requirements.txt"
    assert req_path.exists()
    content = req_path.read_text(encoding="utf-8")
    required_packages = [
        "streamlit",
        "tensorflow",
        "keras",
        "pydicom",
        "requests",
        "pillow",
        "numpy",
        "opencv-python-headless",
    ]
    for pkg in required_packages:
        assert pkg in content, f"Package '{pkg}' missing from requirements.txt"


def test_get_configured_model_url_from_env():
    """Verify MODEL_URL and MEDVISION_MODEL_URL environment variables are correctly retrieved."""
    with patch.dict(os.environ, {"MODEL_URL": "https://example.com/model.keras"}, clear=False):
        assert get_configured_model_url() == "https://example.com/model.keras"

    with patch.dict(os.environ, {"MEDVISION_MODEL_URL": "https://example.com/medvision.keras", "MODEL_URL": ""}, clear=False):
        assert get_configured_model_url() == "https://example.com/medvision.keras"


def test_resolve_model_checkpoint_with_missing_model_raises_actionable_error(tmp_path):
    """Verify resolve_model_checkpoint raises FileNotFoundError with deployment instructions if absent."""
    non_existent_path = tmp_path / "non_existent.keras"
    with patch.dict(os.environ, {"MODEL_URL": "", "MEDVISION_MODEL_URL": "", "MODEL_PATH": ""}, clear=False):
        with patch("app.services.inference_service.Path.home", return_value=tmp_path):
            with pytest.raises(FileNotFoundError) as exc_info:
                # Force search in empty directory mimicking fresh git clone on Streamlit Cloud
                resolve_model_checkpoint(checkpoint_path=non_existent_path, root_dir=tmp_path)
            err_msg = str(exc_info.value)
            assert "MODEL_URL" in err_msg or "MEDVISION_MODEL_URL" in err_msg
            assert "densenet121_stage2_best.keras" in err_msg


def test_download_model_checkpoint_atomic(tmp_path):
    """Verify download_model_checkpoint downloads and writes atomically."""
    mock_url = "https://mock-storage.com/model.keras"
    fake_content = b"FAKE_KERAS_MODEL_BINARY_STREAM_FOR_TESTING"
    dest_file = tmp_path / "test_downloaded.keras"

    mock_response = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[fake_content[:10], fake_content[10:]])
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        result_path = download_model_checkpoint(mock_url, dest_file)
        assert result_path == dest_file
        assert dest_file.exists()
        assert dest_file.read_bytes() == fake_content
        # Temporary file should no longer exist
        assert not (tmp_path / "test_downloaded.keras.tmp").exists()


def test_download_model_checkpoint_cleanup_on_failure(tmp_path):
    """Verify temp files are removed if download fails."""
    mock_url = "https://mock-storage.com/failing_model.keras"
    dest_file = tmp_path / "failing_download.keras"

    with patch("requests.get", side_effect=RuntimeError("Network Timeout")):
        with pytest.raises(RuntimeError) as exc_info:
            download_model_checkpoint(mock_url, dest_file)
        assert "Failed to download model weights" in str(exc_info.value)
        assert not dest_file.exists()
        assert not (tmp_path / "failing_download.keras.tmp").exists()
