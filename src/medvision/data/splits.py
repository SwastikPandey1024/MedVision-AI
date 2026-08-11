"""Patient-aware dataset splitting and leakage verification engine."""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from medvision.utils.logger import get_logger

logger = get_logger("medvision.data.splits")


def verify_zero_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_col: str = "patient_id",
) -> Dict[str, Any]:
    """Audit train/val/test splits to verify zero patient ID overlap.

    Args:
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Test set DataFrame.
        patient_col: Column name containing patient identifiers.

    Returns:
        Dict containing leakage status boolean and overlap patient sets.
    """
    train_patients = set(train_df[patient_col].unique())
    val_patients = set(val_df[patient_col].unique())
    test_patients = set(test_df[patient_col].unique())

    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)

    total_leakage = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    has_zero_leakage = total_leakage == 0

    if not has_zero_leakage:
        logger.error(
            f"PATIENT LEAKAGE DETECTED! Train/Val overlap: {len(train_val_overlap)}, "
            f"Train/Test overlap: {len(train_test_overlap)}, Val/Test overlap: {len(val_test_overlap)}"
        )
    else:
        logger.info("Patient Leakage Audit PASSED: 0% patient overlap across all splits.")

    return {
        "has_zero_leakage": has_zero_leakage,
        "train_patients_count": len(train_patients),
        "val_patients_count": len(val_patients),
        "test_patients_count": len(test_patients),
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
    }


def create_patient_aware_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    target_col: str = "target",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform group-based patient-aware train/val/test split with target stratification.

    Args:
        df: Metadata DataFrame (1 row per unique patient).
        patient_col: Patient ID column name.
        target_col: Classification target column name.
        train_ratio: Ratio for training set.
        val_ratio: Ratio for validation set.
        test_ratio: Ratio for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios (train + val + test) must sum to 1.0")

    # In 1-to-1 manifest, unique patient IDs equal total rows
    val_test_ratio = val_ratio + test_ratio
    relative_test_ratio = test_ratio / val_test_ratio

    # First split: Train vs (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df[target_col],
        random_state=seed,
    )

    # Second split: Val vs Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df[target_col],
        random_state=seed,
    )

    # Run leakage verification assertion
    audit = verify_zero_patient_leakage(train_df, val_df, test_df, patient_col=patient_col)
    if not audit["has_zero_leakage"]:
        raise ValueError("Patient leakage detected during split creation!")

    logger.info(
        f"Splits created cleanly: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )

    return train_df.copy(), val_df.copy(), test_df.copy()


def create_development_subset(
    df: pd.DataFrame,
    sample_fraction: float = 0.05,
    target_col: str = "target",
    seed: int = 42,
) -> pd.DataFrame:
    """Create stratified development subset for fast CPU testing.

    Args:
        df: Input manifest DataFrame.
        sample_fraction: Fraction of dataset to sample (0.0 to 1.0).
        target_col: Stratification target column.
        seed: Random seed integer.

    Returns:
        Sampled DataFrame.
    """
    if sample_fraction >= 1.0:
        return df.copy()

    dev_df, _ = train_test_split(
        df,
        train_size=sample_fraction,
        stratify=df[target_col],
        random_state=seed,
    )

    logger.info(f"Development subset created: {len(dev_df)} samples ({sample_fraction:.1%})")
    return dev_df.copy()
