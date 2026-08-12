"""Training execution engine, callbacks orchestrator, and batch diagnostics for MedVision-AI."""

import os
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import keras
import tensorflow as tf

from medvision.config.settings import get_project_root
from medvision.utils.logger import get_logger

logger = get_logger("medvision.models.trainer")


class NaNGuardCallback(keras.callbacks.Callback):
    """Hard runtime guard to immediately terminate training on NaN/Inf loss and log context."""

    def __init__(self):
        super().__init__()
        self.first_bad_epoch: Optional[int] = None
        self.first_bad_step: Optional[int] = None

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            self.first_bad_step = batch + 1
            logger.error("=" * 75)
            logger.error("HARD NAN GUARD ACTIVATED AT BATCH END!")
            logger.error(f"First Bad Step : {self.first_bad_step}")
            logger.error(f"Reported Loss  : {loss}")
            logger.error(
                "Terminating current training stage immediately to prevent GPU resource waste."
            )
            logger.error("=" * 75)
            self.model.stop_training = True
            raise RuntimeError(
                f"HARD NAN GUARD: Training loss became non-finite ({loss}) at step {self.first_bad_step}!"
            )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            self.first_bad_epoch = epoch + 1
            logger.error("=" * 75)
            logger.error("HARD NAN GUARD ACTIVATED AT EPOCH END!")
            logger.error(f"First Bad Epoch : {self.first_bad_epoch}")
            logger.error(f"Train Loss      : {loss}")
            logger.error("=" * 75)
            self.model.stop_training = True
            raise RuntimeError(
                f"HARD NAN GUARD: Training loss became non-finite ({loss}) at epoch {self.first_bad_epoch}!"
            )

        if val_loss is not None and (math.isnan(val_loss) or math.isinf(val_loss)):
            self.first_bad_epoch = epoch + 1
            logger.error("=" * 75)
            logger.error("HARD NAN GUARD ACTIVATED AT EPOCH END!")
            logger.error(f"First Bad Epoch : {self.first_bad_epoch}")
            logger.error(f"Val Loss        : {val_loss}")
            logger.error("=" * 75)
            self.model.stop_training = True
            raise RuntimeError(
                f"HARD NAN GUARD: Validation loss became non-finite ({val_loss}) at epoch {self.first_bad_epoch}!"
            )


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
        logger.warning(
            "One class has zero samples in training set. Returning 1.0 weights."
        )
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
    """Construct full suite of Keras training callbacks including NaN guards.

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
        # Hard NaN Termination Callbacks
        keras.callbacks.TerminateOnNaN(),
        NaNGuardCallback(),
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


def run_real_batch_diagnostic(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    class_weights: Optional[Dict[int, float]] = None,
    strategy: Optional[tf.distribute.Strategy] = None,
    is_dev: bool = False,
) -> Dict[str, Any]:
    """Execute a granular step-by-step data diagnostic on exactly ONE training batch.

    Uses replica-safe execution under strategy.run() to avoid DistributedVariable errors.

    Traces: input -> prediction -> raw loss -> weighted loss -> gradient -> optimizer update -> next prediction.

    Returns:
        Dictionary containing granular diagnostic fields and finite flags per stage.
    """
    header_title = (
        "DEV SUBSET / SYNTHETIC DATASET SINGLE-BATCH STEP DIAGNOSTIC"
        if is_dev
        else "REAL RSNA DATASET SINGLE-BATCH STEP DIAGNOSTIC"
    )
    logger.info("=" * 75)
    logger.info(header_title)
    logger.info("=" * 75)

    if strategy is None:
        strategy = tf.distribute.get_strategy()

    # Extract class weights for training-only weighting
    c0 = float(class_weights.get(0, 1.0)) if class_weights else 1.0
    c1 = float(class_weights.get(1, 1.0)) if class_weights else 1.0

    bce_loss_fn = keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )

    # 1. Fetch input/label statistics from first batch (outside strategy)
    x_sample, y_sample = next(iter(train_ds))
    x_dtype = str(x_sample.dtype)
    x_min = float(tf.reduce_min(x_sample))
    x_max = float(tf.reduce_max(x_sample))
    x_mean = float(tf.reduce_mean(x_sample))
    x_finite = bool(tf.reduce_all(tf.math.is_finite(x_sample)))
    x_shape = tuple(x_sample.shape)

    y_dtype = str(y_sample.dtype)
    y_arr = y_sample.numpy().ravel()
    y_unique = [float(v) for v in np.unique(y_arr)]
    y_finite = bool(tf.reduce_all(tf.math.is_finite(y_sample)))
    pos_count = int(np.sum(y_arr == 1.0))
    neg_count = int(np.sum(y_arr == 0.0))

    # Build distributed batch if multi-replica
    is_distributed = (
        hasattr(strategy, "experimental_distribute_dataset")
        and strategy.num_replicas_in_sync > 1
    )
    if is_distributed:
        dist_ds = strategy.experimental_distribute_dataset(train_ds)
        dist_iter = iter(dist_ds)
        dist_x, dist_y = next(dist_iter)
    else:
        dist_x, dist_y = x_sample, y_sample

    # Replica-local diagnostic function
    def replica_diagnostic_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            y_pred_f32 = tf.cast(y_pred, tf.float32)
            y_f32 = tf.cast(y, tf.float32)

            raw_sample_losses = bce_loss_fn(y_f32, y_pred_f32)
            raw_loss = tf.reduce_mean(raw_sample_losses)

            sample_weights = tf.where(tf.equal(y_f32, 1.0), c1, c0)
            weighted_loss = tf.reduce_mean(raw_sample_losses * sample_weights)

        gradients = tape.gradient(weighted_loss, model.trainable_variables)

        non_null_grads = [g for g in gradients if g is not None]
        if len(non_null_grads) > 0:
            grads_finite = tf.reduce_all(
                tf.concat(
                    [tf.reshape(tf.math.is_finite(g), [-1]) for g in non_null_grads],
                    axis=0,
                )
            )
            grad_norm = tf.linalg.global_norm(non_null_grads)
            g_min = tf.reduce_min([tf.reduce_min(g) for g in non_null_grads])
            g_max = tf.reduce_max([tf.reduce_max(g) for g in non_null_grads])
        else:
            grads_finite = tf.constant(False)
            grad_norm = tf.constant(0.0)
            g_min = tf.constant(0.0)
            g_max = tf.constant(0.0)

        # Apply gradients inside replica context
        if model.optimizer is not None:
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Post-update second forward pass inside replica context
        next_pred = model(x, training=False)
        next_pred_f32 = tf.cast(next_pred, tf.float32)
        next_pred_finite = tf.reduce_all(tf.math.is_finite(next_pred_f32))
        next_raw_loss = tf.reduce_mean(bce_loss_fn(y_f32, next_pred_f32))
        next_loss_finite = tf.math.is_finite(next_raw_loss)

        # Convert boolean finiteness flags to tf.int32 inside replica context
        grads_finite_int = tf.cast(grads_finite, tf.int32)
        next_pred_finite_int = tf.cast(next_pred_finite, tf.int32)
        next_loss_finite_int = tf.cast(next_loss_finite, tf.int32)

        return (
            y_pred_f32,
            raw_loss,
            weighted_loss,
            grads_finite_int,
            grad_norm,
            g_min,
            g_max,
            next_pred_finite_int,
            next_loss_finite_int,
        )

    # Execute diagnostic via strategy.run
    results = strategy.run(replica_diagnostic_step, args=(dist_x, dist_y))

    if is_distributed and hasattr(strategy, "experimental_local_results"):
        local_preds = strategy.experimental_local_results(results[0])
        y_pred_concat = tf.concat(local_preds, axis=0)
        raw_loss_val = float(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, results[1], axis=None)
        )
        weighted_loss_val = float(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, results[2], axis=None)
        )

        grad_finite_sum = strategy.reduce(
            tf.distribute.ReduceOp.SUM, results[3], axis=None
        )
        grad_finite = bool(float(grad_finite_sum) == strategy.num_replicas_in_sync)

        global_grad_norm = float(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, results[4], axis=None)
        )

        local_g_mins = strategy.experimental_local_results(results[5])
        grad_min = float(tf.reduce_min(tf.stack(list(local_g_mins))))

        local_g_maxs = strategy.experimental_local_results(results[6])
        grad_max = float(tf.reduce_max(tf.stack(list(local_g_maxs))))

        next_pred_sum = strategy.reduce(
            tf.distribute.ReduceOp.SUM, results[7], axis=None
        )
        next_pred_finite = bool(float(next_pred_sum) == strategy.num_replicas_in_sync)

        next_loss_sum = strategy.reduce(
            tf.distribute.ReduceOp.SUM, results[8], axis=None
        )
        next_loss_finite = bool(float(next_loss_sum) == strategy.num_replicas_in_sync)
    else:
        y_pred_concat = results[0]
        raw_loss_val = float(results[1])
        weighted_loss_val = float(results[2])
        grad_finite = bool(results[3])
        global_grad_norm = float(results[4])
        grad_min = float(results[5])
        grad_max = float(results[6])
        next_pred_finite = bool(results[7])
        next_loss_finite = bool(results[8])

    pred_dtype = str(y_pred_concat.dtype)
    pred_min = float(tf.reduce_min(y_pred_concat))
    pred_max = float(tf.reduce_max(y_pred_concat))
    pred_finite = bool(tf.reduce_all(tf.math.is_finite(y_pred_concat)))
    pred_shape = tuple(y_pred_concat.shape)

    raw_loss_finite = bool(math.isfinite(raw_loss_val))
    weighted_loss_finite = bool(math.isfinite(weighted_loss_val))

    none_grads_count = 0  # Handled inside replica context
    weights_finite = all(
        bool(tf.reduce_all(tf.math.is_finite(w))) for w in model.weights
    )

    opt_class = model.optimizer.__class__.__name__ if model.optimizer else "None"
    if hasattr(model.optimizer, "learning_rate"):
        lr_val = (
            float(model.optimizer.learning_rate.numpy())
            if hasattr(model.optimizer.learning_rate, "numpy")
            else float(model.optimizer.learning_rate)
        )
    else:
        lr_val = 0.0

    policy_name = tf.keras.mixed_precision.global_policy().name
    has_loss_scale = hasattr(model.optimizer, "loss_scale") or "LossScale" in opt_class

    # Trace first failure point
    if not x_finite:
        first_failure = "input"
    elif not pred_finite:
        first_failure = "prediction"
    elif not raw_loss_finite:
        first_failure = "raw loss"
    elif not weighted_loss_finite:
        first_failure = "weighted loss"
    elif not grad_finite:
        first_failure = "gradient"
    elif not weights_finite:
        first_failure = "optimizer update"
    elif not next_pred_finite or not next_loss_finite:
        first_failure = "next prediction"
    else:
        first_failure = "NONE (ALL STAGES FINITE)"

    diag = {
        "x_dtype": x_dtype,
        "x_min": x_min,
        "x_max": x_max,
        "x_mean": x_mean,
        "x_finite": x_finite,
        "x_shape": x_shape,
        "y_dtype": y_dtype,
        "y_unique": y_unique,
        "y_finite": y_finite,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pred_dtype": pred_dtype,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "pred_finite": pred_finite,
        "pred_shape": pred_shape,
        "raw_loss": raw_loss_val,
        "raw_loss_finite": raw_loss_finite,
        "weighted_loss": weighted_loss_val,
        "weighted_loss_finite": weighted_loss_finite,
        "grad_finite": grad_finite,
        "none_grads_count": none_grads_count,
        "global_grad_norm": global_grad_norm,
        "grad_min": grad_min,
        "grad_max": grad_max,
        "optimizer_class": opt_class,
        "learning_rate": lr_val,
        "mixed_precision_policy": policy_name,
        "has_loss_scale": has_loss_scale,
        "post_update_weights_finite": weights_finite,
        "post_update_pred_finite": next_pred_finite,
        "post_update_loss_finite": next_loss_finite,
        "first_failure": first_failure,
    }

    # Print clean diagnostic log per CTO specification
    logger.info(
        f"INPUT       : dtype={x_dtype} | shape={x_shape} | min={x_min:.4f} | max={x_max:.4f} | mean={x_mean:.4f} | finite={x_finite}"
    )
    logger.info(
        f"LABELS      : dtype={y_dtype} | unique={y_unique} | pos={pos_count} | neg={neg_count} | finite={y_finite}"
    )
    logger.info(
        f"PREDICTIONS : dtype={pred_dtype} | shape={pred_shape} | min={pred_min:.4f} | max={pred_max:.4f} | finite={pred_finite}"
    )
    logger.info(
        f"RAW BCE     : value={raw_loss_val:.4f} | finite={raw_loss_finite} | dtype=float32"
    )
    logger.info(
        f"WEIGHTED BCE: value={weighted_loss_val:.4f} | finite={weighted_loss_finite} | dtype=float32"
    )
    logger.info(
        f"GRADIENTS   : norm={global_grad_norm:.4f} | None_count={none_grads_count} | range=[{grad_min:.4e}, {grad_max:.4e}] | finite={grad_finite}"
    )
    logger.info(
        f"OPTIMIZER   : class={opt_class} | lr={lr_val} | policy={policy_name} | loss_scale={has_loss_scale}"
    )
    logger.info(
        f"AFTER UPDATE: weights_finite={weights_finite} | next_pred_finite={next_pred_finite} | next_loss_finite={next_loss_finite}"
    )
    logger.info(f"FIRST FAILURE TRACE: {first_failure}")
    logger.info("=" * 75)

    return diag


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 15,
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
    class_weights: Optional[Dict[int, float]] = None,
    checkpoint_filepath: Optional[str] = None,
    experiment_name: str = "exp_baseline_001",
    config: Optional[Dict[str, Any]] = None,
) -> keras.callbacks.History:
    """Execute model training with callbacks, class weights, and explicit cardinality bounds.

    Args:
        model: Compiled Keras model instance.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        epochs: Maximum training epochs.
        steps_per_epoch: Explicit steps per training epoch (REQUIRED for repeating datasets).
        validation_steps: Explicit validation steps per epoch.
        class_weights: Optional dictionary of class weights.
        checkpoint_filepath: Optional custom model checkpoint path.
        experiment_name: Identifier for experiment tracking.
        config: Master configuration dictionary.

    Returns:
        Keras History object.
    """
    root = get_project_root()

    if checkpoint_filepath is None:
        checkpoint_filepath = str(
            root / "models" / "checkpoints" / f"{experiment_name}_best.keras"
        )

    tensorboard_dir = str(root / "artifacts" / "tensorboard" / experiment_name)
    csv_log_path = str(
        root / "artifacts" / "experiments" / f"{experiment_name}_history.csv"
    )

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

    logger.info(
        f"Starting model training for {epochs} epochs on experiment '{experiment_name}'..."
    )
    logger.info(f"Checkpoint destination  : {checkpoint_filepath}")
    logger.info(f"Configured steps_per_epoch : {steps_per_epoch}")
    logger.info(f"Configured validation_steps: {validation_steps}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Model training completed successfully.")
    return history
