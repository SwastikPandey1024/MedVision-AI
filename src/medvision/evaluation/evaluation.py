"""Comprehensive model evaluation and visualization engine for MedVision-AI."""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import keras
import tensorflow as tf

from medvision.utils.logger import get_logger

logger = get_logger("medvision.evaluation")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive clinical evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels array (N, 1) or (N,).
        y_pred_prob: Predicted probability array (N, 1) or (N,).
        threshold: Decision threshold (default 0.5).

    Returns:
        Dictionary of computed metrics.
    """
    y_true = np.array(y_true).ravel()
    y_pred_prob = np.array(y_pred_prob).ravel()
    y_pred_binary = (y_pred_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Sensitivity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    try:
        roc_auc_val = float(auc(*roc_curve(y_true, y_pred_prob)[:2]))
    except Exception:
        roc_auc_val = 0.5

    try:
        pr_auc_val = float(average_precision_score(y_true, y_pred_prob))
    except Exception:
        pr_auc_val = 0.0

    metrics = {
        "threshold": threshold,
        "sample_count": len(y_true),
        "positive_count": int(np.sum(y_true)),
        "negative_count": int(len(y_true) - np.sum(y_true)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1_score, 4),
        "roc_auc": round(roc_auc_val, 4),
        "pr_auc": round(pr_auc_val, 4),
    }

    return metrics


def plot_evaluation_curves(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    output_dir: Path,
    prefix: str = "eval",
) -> Dict[str, Path]:
    """Plot and save Precision-Recall Curve, ROC Curve, and Confusion Matrix Heatmap.

    Args:
        y_true: Ground truth binary labels.
        y_pred_prob: Predicted probability array.
        output_dir: Output directory path.
        prefix: Filename prefix.

    Returns:
        Dictionary of saved plot paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true).ravel()
    y_pred_prob = np.array(y_pred_prob).ravel()

    plot_paths = {}

    # 1. Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc_val = average_precision_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, color="#1f77b4", lw=2.5, label=f"PR Curve (AUC = {pr_auc_val:.4f})")
    ax.set_xlabel("Recall / Sensitivity", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title(f"Precision-Recall Curve ({prefix})", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    pr_svg = output_dir / f"{prefix}_pr_curve.svg"
    pr_png = output_dir / f"{prefix}_pr_curve.png"
    plt.savefig(pr_svg, format="svg", bbox_inches="tight")
    plt.savefig(pr_png, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["pr_curve_svg"] = pr_svg
    plot_paths["pr_curve_png"] = pr_png

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#ff7f0e", lw=2.5, label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title(f"Receiver Operating Characteristic (ROC) Curve ({prefix})", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    roc_svg = output_dir / f"{prefix}_roc_curve.svg"
    roc_png = output_dir / f"{prefix}_roc_curve.png"
    plt.savefig(roc_svg, format="svg", bbox_inches="tight")
    plt.savefig(roc_png, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["roc_curve_svg"] = roc_svg
    plot_paths["roc_curve_png"] = roc_png

    # 3. Confusion Matrix Heatmap
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14, fontweight="bold", color="black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Pneumonia (1)"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["Normal (0)", "Pneumonia (1)"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix ({prefix})", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    cm_svg = output_dir / f"{prefix}_confusion_matrix.svg"
    cm_png = output_dir / f"{prefix}_confusion_matrix.png"
    plt.savefig(cm_svg, format="svg", bbox_inches="tight")
    plt.savefig(cm_png, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["confusion_matrix_svg"] = cm_svg
    plot_paths["confusion_matrix_png"] = cm_png

    logger.info(f"Generated evaluation plots in: {output_dir}")
    return plot_paths


def evaluate_model_performance(
    model: keras.Model,
    dataset: tf.data.Dataset,
    output_dir: Path,
    prefix: str = "eval",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run end-to-end evaluation on dataset, computing metrics and generating curve plots.

    Args:
        model: Trained Keras Model instance.
        dataset: Evaluation tf.data.Dataset yielding (images, labels).
        output_dir: Target output directory for evaluation reports.
        prefix: Filename prefix identifier.
        threshold: Classification decision threshold.

    Returns:
        Evaluation dictionary including metrics and plot artifact paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_list = []
    y_pred_list = []

    logger.info(f"Evaluating model '{model.name}' on evaluation dataset...")

    for x_batch, y_batch in dataset:
        preds = model(x_batch, training=False)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(preds.numpy())

    y_true = np.vstack(y_true_list).ravel()
    y_pred_prob = np.vstack(y_pred_list).ravel()

    metrics = compute_classification_metrics(y_true, y_pred_prob, threshold=threshold)
    plot_paths = plot_evaluation_curves(y_true, y_pred_prob, output_dir, prefix=prefix)

    # Save JSON Report
    json_path = output_dir / f"{prefix}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save Markdown Summary
    md_path = output_dir / f"{prefix}_report.md"
    md_lines = [
        f"# Evaluation Report: {prefix.upper()}",
        f"**Model Identifier:** `{model.name}`  ",
        f"**Total Samples:** `{metrics['sample_count']}`  ",
        f"**Decision Threshold:** `{metrics['threshold']}`  ",
        "",
        "## Performance Metrics Summary",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **PR-AUC (Primary)** | `{metrics['pr_auc']}` |",
        f"| **ROC-AUC** | `{metrics['roc_auc']}` |",
        f"| **F1-Score** | `{metrics['f1_score']}` |",
        f"| **Recall / Sensitivity** | `{metrics['recall_sensitivity']}` |",
        f"| **Specificity** | `{metrics['specificity']}` |",
        f"| **Precision** | `{metrics['precision']}` |",
        f"| **Accuracy** | `{metrics['accuracy']}` |",
        "",
        "## Confusion Matrix Breakdown",
        f"- **True Negatives (TN):** `{metrics['tn']}`",
        f"- **False Positives (FP):** `{metrics['fp']}`",
        f"- **False Negatives (FN):** `{metrics['fn']}`",
        f"- **True Positives (TP):** `{metrics['tp']}`",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    result = {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred_prob": y_pred_prob,
        "json_path": json_path,
        "md_path": md_path,
        "plots": plot_paths,
    }

    logger.info(f"Evaluation complete. PR-AUC: {metrics['pr_auc']} | ROC-AUC: {metrics['roc_auc']}")
    return result
