"""Validation-based decision threshold selection engine for MedVision-AI.

CRITICAL MLOPS / CLINICAL RULE:
Threshold selection MUST use ONLY validation set predictions.
Test labels are strictly excluded during threshold optimization to prevent test-set leakage.
"""

from typing import Dict, Any
import numpy as np
from medvision.evaluation.evaluation import compute_classification_metrics
from medvision.utils.logger import get_logger

logger = get_logger("medvision.evaluation.threshold")


def select_optimal_threshold_from_val(
    val_y_true: np.ndarray,
    val_y_pred_prob: np.ndarray,
    criterion: str = "f1_score",
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    step: float = 0.01,
) -> Dict[str, Any]:
    """Select optimal classification decision threshold using VALIDATION predictions ONLY.

    Args:
        val_y_true: Validation ground truth binary labels array (N,).
        val_y_pred_prob: Validation predicted probability array (N,).
        criterion: Primary metric to maximize ('f1_score', 'accuracy', 'pr_auc', 'youden_j').
        min_threshold: Minimum threshold candidate.
        max_threshold: Maximum threshold candidate.
        step: Search step size.

    Returns:
        Dict containing selected_threshold, best_val_score, and threshold audit details.
    """
    val_y_true = np.array(val_y_true).ravel()
    val_y_pred_prob = np.array(val_y_pred_prob).ravel()

    thresholds = np.arange(min_threshold, max_threshold + step / 2.0, step)
    best_threshold = 0.5
    best_score = -1.0
    scores_history = []

    for th in thresholds:
        metrics = compute_classification_metrics(val_y_true, val_y_pred_prob, threshold=float(th))
        if criterion == "youden_j":
            score = metrics["recall_sensitivity"] + metrics["specificity"] - 1.0
        else:
            score = metrics.get(criterion, metrics["f1_score"])

        scores_history.append((float(th), float(score)))
        if score > best_score:
            best_score = score
            best_threshold = float(th)

    logger.info(
        f"Selected optimal threshold on VALIDATION data: {best_threshold:.4f} "
        f"(Validation {criterion} = {best_score:.4f}). TEST DATA WAS NOT USED."
    )

    return {
        "selected_threshold": round(best_threshold, 4),
        "criterion": criterion,
        "best_val_score": round(best_score, 4),
        "test_data_used": False,
        "scores_history": scores_history,
    }
