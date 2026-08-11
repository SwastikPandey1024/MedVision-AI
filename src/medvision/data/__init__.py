"""Data ingestion, validation, splitting, and EDA modules."""

from medvision.data.dataset import find_dataset_root, parse_rsna_manifest
from medvision.data.eda import generate_eda_report
from medvision.data.splits import (
    create_development_subset,
    create_patient_aware_splits,
    verify_zero_patient_leakage,
)
from medvision.data.validation import validate_manifest_integrity

__all__ = [
    "find_dataset_root",
    "parse_rsna_manifest",
    "validate_manifest_integrity",
    "create_patient_aware_splits",
    "verify_zero_patient_leakage",
    "create_development_subset",
    "generate_eda_report",
]
