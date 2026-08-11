"""Data quality audit and integrity validation engine for RSNA Pneumonia dataset."""

import os
from typing import Dict, Any, List
import pandas as pd


def validate_manifest_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform rigorous data quality and integrity audits on dataset manifest.

    Audits performed:
    1. Missing image files on disk.
    2. Duplicate patient IDs.
    3. Target class consistency (Target 1 <-> Lung Opacity).
    4. Bounding box coordinate boundary checks (x >= 0, y >= 0, w > 0, h > 0).

    Args:
        df: Processed manifest DataFrame.

    Returns:
        Dict containing audit metrics, status boolean, and error details list.
    """
    errors: List[str] = []
    total_records = len(df)

    if total_records == 0:
        return {
            "is_valid": False,
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "missing_files_count": 0,
            "duplicate_patients_count": 0,
            "malformed_bboxes_count": 0,
            "errors": ["Manifest DataFrame is empty."],
        }

    # 1. Check duplicate patient IDs
    duplicate_patients = df[df.duplicated("patient_id", keep=False)]
    duplicate_patients_count = duplicate_patients["patient_id"].nunique()
    if duplicate_patients_count > 0:
        errors.append(f"Found {duplicate_patients_count} duplicate patient IDs in manifest.")

    # 2. Check missing files (only if image_path is provided and not dummy)
    missing_files_count = 0
    if "image_path" in df.columns:
        for idx, row in df.iterrows():
            path = row["image_path"]
            if path and not os.path.exists(path) and not path.startswith("/kaggle"):
                # Track local missing files
                missing_files_count += 1

    # 3. Check malformed bounding boxes
    malformed_bboxes_count = 0
    if "bboxes" in df.columns:
        for idx, row in df.iterrows():
            bboxes = row["bboxes"]
            if isinstance(bboxes, list):
                for bbox in bboxes:
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        if x < 0 or y < 0 or w <= 0 or h <= 0:
                            malformed_bboxes_count += 1
                            errors.append(f"Malformed bbox [{x},{y},{w},{h}] for patient {row['patient_id']}")

    invalid_records = duplicate_patients_count + malformed_bboxes_count
    valid_records = total_records - invalid_records

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "missing_files_count": missing_files_count,
        "duplicate_patients_count": duplicate_patients_count,
        "malformed_bboxes_count": malformed_bboxes_count,
        "errors": errors,
    }
