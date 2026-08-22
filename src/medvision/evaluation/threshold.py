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


def save_threshold_audit_report(
    audit_result: Dict[str, Any],
    output_dir: Any,
    prefix: str = "threshold",
) -> Dict[str, Any]:
    """Persist threshold selection audit in JSON and Markdown formats.

    Args:
        audit_result: Dict returned by select_optimal_threshold_from_val.
        output_dir: Target directory path.
        prefix: Filename prefix identifier.

    Returns:
        Dict mapping artifact keys ('json_path', 'md_path') to Path objects.
    """
    from pathlib import Path
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{prefix}_selection_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    selected_th = audit_result.get("selected_threshold", 0.50)
    best_score = audit_result.get("best_val_score", 0.0)
    crit = audit_result.get("criterion", "f1_score")

    md_path = output_dir / f"{prefix}_selection_audit.md"
    md_lines = [
        f"# Decision Threshold Optimization Audit: {prefix.upper()}",
        "",
        "> [!IMPORTANT]",
        "> **Strict Zero-Leakage Protocol**: Threshold optimization was performed **exclusively on validation set predictions**.",
        "> Test labels and predictions were strictly excluded during threshold search to prevent data snooping.",
        "",
        "## Summary",
        f"- **Optimized Criterion:** `{crit}`",
        f"- **Selected Optimal Threshold:** `{selected_th:.4f}`",
        f"- **Validation Score at Optimal Threshold:** `{best_score:.4f}`",
        f"- **Test Data Used:** `{audit_result.get('test_data_used', False)}` (Strictly Protected)",
        "",
        "## Threshold Scan History (Candidate Progression)",
        "| Threshold Candidate | Validation Score | Status |",
        "| :--- | :--- | :--- |",
    ]

    scores = audit_result.get("scores_history", [])
    for th, sc in scores:
        if np.isclose(th % 0.05, 0.0) or np.isclose(th, selected_th, atol=1e-4):
            marker = "⭐ **Selected Optimal**" if np.isclose(th, selected_th, atol=1e-4) else "Candidate"
            md_lines.append(f"| `{th:.2f}` | `{sc:.4f}` | {marker} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Saved threshold audit reports to {json_path} and {md_path}")
    return {"json_path": json_path, "md_path": md_path}
