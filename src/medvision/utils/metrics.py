"""Evaluation metrics module for MedVision-AI.

Provides Keras 3 metrics tracking:
- loss
- accuracy
- precision
- recall (sensitivity)
- specificity
- f1_score
- roc_auc
- pr_auc (Primary metric for checkpointing due to ~22.5% class imbalance)
"""

from typing import List
import keras
from keras import ops


@keras.saving.register_keras_serializable(package="medvision.metrics")
class Specificity(keras.metrics.Metric):
    """Custom Keras metric to calculate Specificity (True Negative Rate = TN / (TN + FP))."""

    def __init__(self, name: str = "specificity", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred > self.threshold, "float32")

        # True Negatives: y_true == 0 AND y_pred == 0
        tn = ops.sum((1.0 - y_true) * (1.0 - y_pred))
        # False Positives: y_true == 0 AND y_pred == 1
        fp = ops.sum((1.0 - y_true) * y_pred)

        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)

    def result(self):
        denominator = self.true_negatives + self.false_positives
        return ops.divide_no_nan(self.true_negatives, denominator)

    def reset_state(self):
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


@keras.saving.register_keras_serializable(package="medvision.metrics")
class F1Score(keras.metrics.Metric):
    """Custom Keras metric to calculate F1 Score = 2 * (Precision * Recall) / (Precision + Recall)."""

    def __init__(self, name: str = "f1_score", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision_metric = keras.metrics.Precision(thresholds=threshold)
        self.recall_metric = keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision_metric.result()
        r = self.recall_metric.result()
        return ops.divide_no_nan(2.0 * p * r, p + r)

    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


def get_model_metrics() -> List[keras.metrics.Metric]:
    """Get standard set of evaluation metrics required for MedVision-AI models.

    Returns:
        List of initialized Keras Metric instances.
    """
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        Specificity(name="specificity"),
        F1Score(name="f1_score"),
        keras.metrics.AUC(curve="ROC", name="roc_auc"),
        keras.metrics.AUC(curve="PR", name="pr_auc"),
    ]
