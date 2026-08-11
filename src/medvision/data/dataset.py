"""Dataset ingestion and tf.data pipeline creation placeholder (Phase 1 & 2)."""

from typing import Tuple
import pandas as pd
import tensorflow as tf


def load_dataset_metadata(csv_path: str) -> pd.DataFrame:
    """Load metadata index CSV for RSNA or general X-ray dataset.

    Args:
        csv_path: Path to dataset manifest CSV file.

    Returns:
        Pandas DataFrame containing patient IDs, image paths, and labels.
    """
    raise NotImplementedError("Dataset ingestion will be implemented in Phase 1.")


def create_tf_dataset(
    df: pd.DataFrame,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Build high-performance tf.data pipeline for model training/evaluation.

    Args:
        df: Metadata DataFrame containing image paths and labels.
        image_size: Target tuple (height, width).
        batch_size: Number of samples per batch.
        is_training: Whether to enable shuffling and augmentation.

    Returns:
        Configured tf.data.Dataset object.
    """
    raise NotImplementedError("tf.data input pipeline will be implemented in Phase 2.")
