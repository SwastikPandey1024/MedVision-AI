"""Patient-aware dataset splitting to prevent data leakage (Phase 1)."""

from typing import Tuple
import pandas as pd


def create_patient_aware_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform group-based patient-aware train/val/test split.

    Args:
        df: Metadata DataFrame with patient identifier column.
        patient_col: Name of column containing unique patient identifiers.
        train_ratio: Percentage for training set.
        val_ratio: Percentage for validation set.
        test_ratio: Percentage for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    raise NotImplementedError("Patient-aware splitting will be implemented in Phase 1.")
