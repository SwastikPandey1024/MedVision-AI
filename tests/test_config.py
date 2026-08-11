"""Unit tests for configuration loader and settings."""

from medvision.config.settings import load_config, get_project_root


def test_get_project_root():
    """Verify project root path exists."""
    root = get_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_load_config(sample_config):
    """Verify master configuration loading parameters."""
    assert "project" in sample_config
    assert "execution" in sample_config
    assert "device" in sample_config
    assert "data" in sample_config
    assert "model" in sample_config
    assert sample_config["project"]["name"] == "MedVision-AI"
    assert sample_config["execution"]["mode"] in ("development", "full")
    assert sample_config["model"]["selected_architecture"] in ("custom_cnn", "densenet121", "efficientnetb0")
