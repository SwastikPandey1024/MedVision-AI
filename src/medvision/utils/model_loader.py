"""Model loading and serialization compatibility utilities for MedVision-AI."""

from pathlib import Path
from typing import Union, Optional, Any, Dict
from contextlib import contextmanager
import keras
from medvision.utils.logger import get_logger

logger = get_logger("medvision.utils.model_loader")


@contextmanager
def _scoped_deserialization_compatibility():
    """Temporarily install compatibility handlers during deserialization and restore immediately."""
    orig_dense_from_config = keras.layers.Dense.from_config

    def compatible_dense_from_config(cls: Any, config: Dict[str, Any]) -> keras.layers.Dense:
        if isinstance(config, dict):
            config = dict(config)
            config.pop("quantization_config", None)
        return orig_dense_from_config(config)

    keras.layers.Dense.from_config = classmethod(compatible_dense_from_config)  # type: ignore
    try:
        yield
    finally:
        keras.layers.Dense.from_config = orig_dense_from_config


def load_medvision_model(
    checkpoint_path: Union[str, Path],
    compile: bool = False,
    safe_mode: bool = False,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Load a trained MedVision-AI Keras model checkpoint safely with isolated cross-version compatibility.

    Args:
        checkpoint_path: Filepath to the .keras model archive.
        compile: Whether to compile the model after loading.
        safe_mode: Whether to run deserialization in safe mode.
        custom_objects: Optional dict of custom layers or metrics.

    Returns:
        Loaded Keras Model.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path.resolve()}")

    logger.info(f"Loading MedVision model from: {path.resolve()}")
    with _scoped_deserialization_compatibility():
        model = keras.models.load_model(
            path,
            custom_objects=custom_objects,
            compile=compile,
            safe_mode=safe_mode,
        )
    return model
