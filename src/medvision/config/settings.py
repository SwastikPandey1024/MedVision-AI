"""Settings loader and validation utility."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file and validate critical fields.

    Args:
        config_path: Path to YAML config file. If None, loads default config.yaml.

    Returns:
        Dict containing configuration parameters.
    """
    if config_path is None:
        root = get_project_root()
        config_path = str(root / "src" / "medvision" / "config" / "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Validate execution mode
    execution_mode = config.get("execution", {}).get("mode", "development")
    if execution_mode not in ("development", "full"):
        raise ValueError(f"Invalid execution mode: '{execution_mode}'. Must be 'development' or 'full'.")

    # Validate architecture selection
    arch = config.get("model", {}).get("selected_architecture", "densenet121")
    valid_archs = ("custom_cnn", "densenet121", "efficientnetb0")
    if arch not in valid_archs:
        raise ValueError(f"Invalid architecture: '{arch}'. Must be one of {valid_archs}.")

    return config


def get_output_base_dir() -> Path:
    """Return the absolute output directory path for training artifacts outside Git repo.

    Defaults to `/kaggle/working/medvision_outputs` if running on Kaggle or if MEDVISION_OUTPUT_DIR env var is set.
    """
    env_dir = os.environ.get("MEDVISION_OUTPUT_DIR")
    if env_dir:
        base_path = Path(env_dir)
    elif os.path.exists("/kaggle/working"):
        base_path = Path("/kaggle/working/medvision_outputs")
    else:
        base_path = Path("/kaggle/working/medvision_outputs")
    return base_path


def get_output_dir(subfolder: str = "") -> Path:
    """Return and automatically create a specific output subfolder (checkpoints, logs, metrics)."""
    base = get_output_base_dir()
    dir_path = base / subfolder if subfolder else base
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

