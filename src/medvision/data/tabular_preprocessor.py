"""Tabular data preprocessing pipeline module using scikit-learn."""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from medvision.utils.logger import get_logger

logger = get_logger("medvision.data.tabular_preprocessor")


def build_tabular_preprocessor(
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer preprocessor for tabular features.

    Numerical features: Imputed with median and scaled using StandardScaler.
    Categorical features: Imputed with most frequent and encoded using OneHotEncoder.

    Args:
        numeric_features: List of numerical column names.
        categorical_features: Optional list of categorical column names.

    Returns:
        Configured scikit-learn ColumnTransformer object.
    """
    if categorical_features is None:
        categorical_features = []

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("num", numeric_transformer, numeric_features)]

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str | Path) -> str:
    """Serialize and save fitted ColumnTransformer preprocessor object to disk using joblib.

    Args:
        preprocessor: Fitted ColumnTransformer object.
        filepath: Output path string or Path object.

    Returns:
        Path string where preprocessor was saved.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, filepath)
    logger.info(f"Saved tabular preprocessor to {filepath}")
    return str(filepath)


def load_preprocessor(filepath: str | Path) -> ColumnTransformer:
    """Load serialized ColumnTransformer preprocessor object from disk using joblib.

    Args:
        filepath: Path string or Path object.
    Returns:
        Loaded ColumnTransformer object.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
    preprocessor = joblib.load(filepath)
    logger.info(f"Loaded tabular preprocessor from {filepath}")
    return preprocessor
