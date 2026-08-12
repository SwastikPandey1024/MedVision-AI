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
