"""Unit tests for Phase 1 data ingestion, validation, splitting, and EDA modules."""

import pytest
import pandas as pd
from medvision.data.validation import validate_manifest_integrity
from medvision.data.splits import (
    create_patient_aware_splits,
    verify_zero_patient_leakage,
    create_development_subset,
)
from medvision.data.eda import generate_eda_report


@pytest.fixture
def synthetic_manifest():
    """Fixture generating a synthetic RSNA-style patient manifest DataFrame."""
    records = []
    # Generate 100 unique patients
    for i in range(100):
        patient_id = f"patient_{i:03d}"
        target = 1 if i < 30 else 0
        detailed_class = "Lung Opacity" if target == 1 else ("Normal" if i >= 65 else "No Lung Opacity / Not Normal")
        bboxes = [[100.0, 150.0, 50.0, 60.0]] if target == 1 else []
        records.append({
            "patient_id": patient_id,
            "target": target,
            "detailed_class": detailed_class,
            "bbox_count": len(bboxes),
            "bboxes": bboxes,
            "image_path": f"/tmp/images/{patient_id}.png",
        })
    return pd.DataFrame(records)


def test_validate_manifest_integrity(synthetic_manifest):
    """Test validation engine on synthetic clean manifest."""
    res = validate_manifest_integrity(synthetic_manifest)
    assert res["total_records"] == 100
    assert res["duplicate_patients_count"] == 0
    assert res["malformed_bboxes_count"] == 0


def test_patient_aware_splits(synthetic_manifest):
    """Test patient-aware train/val/test split and ratios."""
    train_df, val_df, test_df = create_patient_aware_splits(
        synthetic_manifest, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Check zero patient leakage assertion
    audit = verify_zero_patient_leakage(train_df, val_df, test_df)
    assert audit["has_zero_leakage"] is True
    assert audit["train_val_overlap_count"] == 0
    assert audit["train_test_overlap_count"] == 0
    assert audit["val_test_overlap_count"] == 0


def test_development_subset(synthetic_manifest):
    """Test creation of stratified development subset."""
    dev_df = create_development_subset(synthetic_manifest, sample_fraction=0.10, seed=42)
    assert len(dev_df) == 10
    # Check target stratification proportion (~30% positive)
    assert dev_df["target"].sum() == 3


def test_eda_reporting(synthetic_manifest, tmp_path):
    """Test EDA report generator exports JSON and Markdown files."""
    report = generate_eda_report(synthetic_manifest, output_dir=tmp_path)
    assert report["total_unique_patients"] == 100
    assert report["class_distribution"]["positive_pneumonia_count"] == 30
    assert report["class_distribution"]["negative_normal_count"] == 70

    assert (tmp_path / "eda_report.json").exists()
    assert (tmp_path / "eda_report.md").exists()


def test_find_dataset_root_nested_kaggle_paths(tmp_path):
    """Test dynamic path resolution and parsing for nested Kaggle competition input directories."""
    from medvision.data.dataset import parse_rsna_manifest

    dataset_dir = tmp_path / "competitions" / "rsna-pneumonia-detection-challenge"
    dataset_dir.mkdir(parents=True)
    labels_csv = dataset_dir / "stage_2_train_labels.csv"
    class_info_csv = dataset_dir / "stage_2_detailed_class_info.csv"
    labels_csv.write_text("patientId,x,y,width,height,Target\np1,10.0,10.0,20.0,20.0,1\n")
    class_info_csv.write_text("patientId,class\np1,Lung Opacity\n")

    manifest = parse_rsna_manifest(dataset_dir)
    assert len(manifest) == 1
    assert manifest.iloc[0]["patient_id"] == "p1"
    assert manifest.iloc[0]["target"] == 1


