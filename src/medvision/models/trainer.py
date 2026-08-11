"""Training execution engine and callbacks orchestrator for MedVision-AI."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import keras
import tensorflow as tf

from medvision.config.settings import get_project_root
from medvision.utils.logger import get_logger

logger = get_logger("medvision.models.trainer")


def compute_training_class_weights(train_manifest_df: pd.DataFrame) -> Dict[int, float]:
    """Calculate balanced class weights using TRAINING DATA ONLY.

    CRITICAL CTO REQUIREMENT:
    Calculates weights using only training labels. Validation and test labels
    are strictly excluded to prevent data leakage.

    Formula:
        W_0 = N / (2 * N_0)
        W_1 = N / (2 * N_1)

    Args:
        train_manifest_df: Pandas DataFrame containing training set manifest ('target' column).

    Returns:
        Dictionary mapping class_id (0, 1) to weight float values.
    """
    if "target" not in train_manifest_df.columns:
        raise ValueError("train_manifest_df must contain a 'target' column.")

    targets = train_manifest_df["target"].values
    n_samples = len(targets)
    n_neg = np.sum(targets == 0)
    n_pos = np.sum(targets == 1)

    if n_neg == 0 or n_pos == 0:
        logger.warning("One class has zero samples in training set. Returning 1.0 weights.")
        return {0: 1.0, 1: 1.0}

    weight_0 = float(n_samples / (2.0 * n_neg))
    weight_1 = float(n_samples / (2.0 * n_pos))

    class_weights = {0: weight_0, 1: weight_1}
    logger.info(
        f"Computed training class weights (N_train={n_samples}, N_normal={n_neg}, N_pneumonia={n_pos}): "
        f"Class 0 (Normal) = {weight_0:.4f}, Class 1 (Pneumonia) = {weight_1:.4f}"
    )
    return class_weights


def build_callbacks(
    checkpoint_filepath: str,
    tensorboard_dir: str,
    csv_log_path: str,
    monitor_metric: str = "val_pr_auc",
    mode: str = "max",
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 3,
) -> list:
    """Construct full suite of Keras training callbacks.

    CRITICAL CTO REQUIREMENT:
    Uses `val_pr_auc` as primary checkpointing & early stopping metric.

    Args:
        checkpoint_filepath: Destination path for best model checkpoint (.keras).
        tensorboard_dir: Directory for TensorBoard event logs.
        csv_log_path: Path for CSVLogger metric output.
        monitor_metric: Primary metric to monitor ('val_pr_auc').
        mode: Metric optimization mode ('max' for PR-AUC).
        early_stopping_patience: Epoch patience before early stopping.
        reduce_lr_patience: Epoch patience before LR reduction.

    Returns:
        List of initialized Keras Callback objects.
    """
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)

    callbacks = [
        # Save best model based on val_pr_auc
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor=monitor_metric,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        # Early Stopping on val_pr_auc plateau
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR on val_pr_auc plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode=mode,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        # CSV log output
        keras.callbacks.CSVLogger(
            filename=csv_log_path,
            append=True,
        ),
        # TensorBoard output
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
        ),
    ]

    return callbacks


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 15,
    class_weights: Optional[Dict[int, float]] = None,
    checkpoint_filepath: Optional[str] = None,
    experiment_name: str = "exp_baseline_001",
    config: Optional[Dict[str, Any]] = None,
) -> keras.callbacks.History:
    """Execute model training with callbacks and class weights.

    Args:
        model: Compiled Keras model instance.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        epochs: Maximum training epochs.
        class_weights: Optional dictionary of class weights.
        checkpoint_filepath: Optional custom model checkpoint path.
        experiment_name: Identifier for experiment tracking.
        config: Master configuration dictionary.

    Returns:
        Keras History object.
    """
    root = get_project_root()

    if checkpoint_filepath is None:
        checkpoint_filepath = str(root / "models" / "checkpoints" / f"{experiment_name}_best.keras")

    tensorboard_dir = str(root / "artifacts" / "tensorboard" / experiment_name)
    csv_log_path = str(root / "artifacts" / "experiments" / f"{experiment_name}_history.csv")

    patience_es = 5
    patience_lr = 3
    if config and "training" in config:
        patience_es = config["training"].get("early_stopping_patience", 5)
        patience_lr = config["training"].get("reduce_lr_patience", 3)

    callbacks = build_callbacks(
        checkpoint_filepath=checkpoint_filepath,
        tensorboard_dir=tensorboard_dir,
        csv_log_path=csv_log_path,
        monitor_metric="val_pr_auc",
        mode="max",
        early_stopping_patience=patience_es,
        reduce_lr_patience=patience_lr,
    )

    logger.info(f"Starting model training for {epochs} epochs on experiment '{experiment_name}'...")
    logger.info(f"Checkpoint destination: {checkpoint_filepath}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Model training completed successfully.")
    return history
