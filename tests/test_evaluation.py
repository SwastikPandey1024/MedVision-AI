"""Unit tests for Phase 4 evaluation engine and metrics computation."""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

from medvision.evaluation import (
    compute_classification_metrics,
    plot_evaluation_curves,
    evaluate_model_performance,
)
from medvision.models.baseline_cnn import build_custom_cnn


def test_compute_classification_metrics():
    """Verify classification metrics calculation accuracy."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.6, 0.4])

    metrics = compute_classification_metrics(y_true, y_pred, threshold=0.5)

    assert metrics["sample_count"] == 8
    assert metrics["positive_count"] == 4
    assert metrics["negative_count"] == 4
    assert metrics["tp"] == 3
    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.75
    assert metrics["recall_sensitivity"] == 0.75
    assert metrics["specificity"] == 0.75
    assert metrics["precision"] == 0.75
    assert metrics["f1_score"] == 0.75
    assert metrics["roc_auc"] > 0.5
    assert metrics["pr_auc"] > 0.5


def test_plot_evaluation_curves(tmp_path):
    """Verify plotting functions generate SVG and PNG files cleanly."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    out_dir = tmp_path / "plots"
    paths = plot_evaluation_curves(y_true, y_pred, out_dir, prefix="test_run")

    assert paths["pr_curve_svg"].exists() and paths["pr_curve_svg"].stat().st_size > 0
    assert paths["pr_curve_png"].exists() and paths["pr_curve_png"].stat().st_size > 0
    assert paths["roc_curve_svg"].exists() and paths["roc_curve_svg"].stat().st_size > 0
    assert paths["roc_curve_png"].exists() and paths["roc_curve_png"].stat().st_size > 0
    assert paths["confusion_matrix_svg"].exists() and paths["confusion_matrix_svg"].stat().st_size > 0
    assert paths["confusion_matrix_png"].exists() and paths["confusion_matrix_png"].stat().st_size > 0


def test_evaluate_model_performance(tmp_path):
    """Verify end-to-end dataset evaluation on a model."""
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=1)

    dummy_images = np.ones((8, 224, 224, 3), dtype=np.float32)
    dummy_labels = np.array([[0], [0], [0], [0], [1], [1], [1], [1]], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).batch(4)

    out_dir = tmp_path / "eval_output"
    res = evaluate_model_performance(model, ds, out_dir, prefix="dummy_model")

    assert res["json_path"].exists()
    assert res["md_path"].exists()
    assert res["metrics"]["sample_count"] == 8


def test_select_optimal_threshold_from_val():
    """Verify threshold selection uses validation predictions only without test data."""
    from medvision.evaluation import select_optimal_threshold_from_val
    val_y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    val_y_pred = np.array([0.1, 0.2, 0.4, 0.3, 0.6, 0.7, 0.8, 0.9])

    th_result = select_optimal_threshold_from_val(val_y_true, val_y_pred, criterion="f1_score")

    assert "selected_threshold" in th_result
    assert 0.1 <= th_result["selected_threshold"] <= 0.9
    assert th_result["test_data_used"] is False


def test_generate_model_comparison_report(tmp_path):
    """Verify model comparison report generation and PR-AUC primary ranking."""
    from medvision.evaluation import generate_model_comparison_report

    models_data = [
        {"model_name": "Custom CNN Baseline", "pr_auc": 0.7200, "roc_auc": 0.8100, "accuracy": 0.8500, "params": 500000},
        {"model_name": "DenseNet121 Stage 2", "pr_auc": 0.8900, "roc_auc": 0.9400, "accuracy": 0.9100, "params": 7301185},
        {"model_name": "DenseNet121 Stage 1", "pr_auc": 0.8300, "roc_auc": 0.8900, "accuracy": 0.8800, "params": 7301185},
    ]

    out_dir = tmp_path / "comp"
    res = generate_model_comparison_report(models_data, out_dir)

    assert res["markdown_path"].exists()
    assert res["json_path"].exists()

    df = res["comparison_df"]
    # Check that rank #1 is DenseNet121 Stage 2 (highest PR-AUC = 0.8900)
    assert df.iloc[0]["model_name"] == "DenseNet121 Stage 2"
    assert df.iloc[1]["model_name"] == "DenseNet121 Stage 1"
    assert df.iloc[2]["model_name"] == "Custom CNN Baseline"


def test_resolve_evaluation_datasets_development_mode():
    """Verify evaluation dataset resolution in development mode."""
    import sys
    from pathlib import Path
    # Import resolve_evaluation_datasets from scripts.evaluate
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from evaluate import resolve_evaluation_datasets

    mode, datasets, meta = resolve_evaluation_datasets(mode="development", batch_size=8)

    assert mode == "development"
    assert "val" in datasets and "test" in datasets and "train" in datasets
    assert meta["source"] == "dev_subset"
    assert meta["n_val"] > 0
    assert meta["n_test"] > 0

    # Verify we can extract a batch from the validation dataset
    for x_b, y_b in datasets["val"].take(1):
        assert x_b.shape[1:] == (224, 224, 3)
        assert y_b.shape[1:] == (1,)


def test_resolve_evaluation_datasets_auto_fallback():
    """Verify evaluation dataset resolution falls back gracefully in auto mode."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from evaluate import resolve_evaluation_datasets

    mode, datasets, meta = resolve_evaluation_datasets(mode="auto", batch_size=8)
    assert mode in ["development", "full"]
    assert "val" in datasets and "test" in datasets


def test_resolve_evaluation_datasets_full_mode_with_synthetic_manifest(tmp_path):
    """Verify full mode data resolution and patient-aware split pipeline using mock RSNA folder."""
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from evaluate import resolve_evaluation_datasets

    mock_rsna = tmp_path / "rsna"
    mock_rsna.mkdir(parents=True)
    images_dir = mock_rsna / "stage_2_train_images"
    images_dir.mkdir(parents=True)

    # Create dummy labels for 20 patients
    rows = []
    for i in range(20):
        pid = f"PATIENT_{i:03d}"
        target = 1 if i < 6 else 0
        rows.append(f"{pid},100.0,100.0,50.0,50.0,{target}")
        # Create a dummy image file
        (images_dir / f"{pid}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    labels_csv = mock_rsna / "stage_2_train_labels.csv"
    labels_csv.write_text("patientId,x,y,width,height,Target\n" + "\n".join(rows) + "\n")

    mode, datasets, meta = resolve_evaluation_datasets(mode="full", batch_size=4, dataset_dir=mock_rsna)
    assert mode == "full"
    assert meta["source"] == "real_rsna"
    assert meta["n_train"] == 14
    assert meta["n_val"] == 3
    assert meta["n_test"] == 3


def test_evaluate_script_cli_execution(tmp_path, monkeypatch):
    """Verify end-to-end evaluate CLI execution for --split all in development mode."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import evaluate

    # Build and save a dummy model
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=1)
    ckpt_path = tmp_path / "dummy_model.keras"
    model.save(ckpt_path)

    out_dir = tmp_path / "eval_all_output"

    # Simulate command line arguments
    test_args = [
        "evaluate.py",
        "--checkpoint", str(ckpt_path),
        "--mode", "development",
        "--split", "all",
        "--batch-size", "8",
        "--threshold", "0.5",
        "--output-dir", str(out_dir),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    evaluate.main()

    # Check generated files
    assert (out_dir / "dummy_model_val_report.json").exists()
    assert (out_dir / "dummy_model_val_report.md").exists()
    assert (out_dir / "dummy_model_test_report.json").exists()
    assert (out_dir / "dummy_model_test_report.md").exists()
    assert (out_dir / "model_comparison_report.md").exists()
    assert (out_dir / "model_comparison_summary.json").exists()
