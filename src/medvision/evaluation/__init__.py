"""MedVision-AI Evaluation Module."""

from medvision.evaluation.evaluation import (
    compute_classification_metrics,
    plot_evaluation_curves,
    evaluate_model_performance,
)
from medvision.evaluation.threshold import (
    select_optimal_threshold_from_val,
    save_threshold_audit_report,
)
from medvision.evaluation.reporting import generate_model_comparison_report

__all__ = [
    "compute_classification_metrics",
    "plot_evaluation_curves",
    "evaluate_model_performance",
    "select_optimal_threshold_from_val",
    "save_threshold_audit_report",
    "generate_model_comparison_report",
]
