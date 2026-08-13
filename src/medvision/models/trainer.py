"""Training execution engine, callbacks orchestrator, and batch diagnostics for MedVision-AI."""

import os
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import keras
import tensorflow as tf

from medvision.config.settings import get_project_root, get_output_dir
from medvision.utils.logger import get_logger

logger = get_logger("medvision.models.trainer")


def verify_checkpoint_persistence(checkpoint_path: str) -> bool:
    """Verify that a trained model checkpoint exists and has non-zero size.

    Raises:
        RuntimeError: If checkpoint file does not exist or has 0 bytes size.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        err_msg = (
            f"CHECKPOINT PERSISTENCE FAILURE! Model checkpoint file does not exist at: {checkpoint_path}\n"
            f"Training cannot be considered successful."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    size_bytes = p.stat().st_size
    if size_bytes == 0:
        err_msg = (
            f"CHECKPOINT PERSISTENCE FAILURE! Model checkpoint at {checkpoint_path} has 0 bytes (empty file)!\n"
            f"Training cannot be considered successful."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    size_mb = size_bytes / (1024 * 1024)
    print("\n" + "=" * 75)
    print("CHECKPOINT PERSISTENCE: PASS")
    print(f"CHECKPOINT PATH: {p.resolve()}")
    print(f"CHECKPOINT SIZE MB: {size_mb:.2f} MB")
    print("=" * 75 + "\n")
    logger.info(
        f"CHECKPOINT PERSISTENCE: PASS | PATH: {p.resolve()} | SIZE: {size_mb:.2f} MB"
    )
    return True



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


class BatchLossTrackerCallback(keras.callbacks.Callback):
    """Forensic Keras callback that tracks per-batch loss, metrics, weights, and optimizer slot variables.

    Identifies the EXACT first batch and internal state tensor that becomes non-finite during model.fit().
    """

    def __init__(self, steps_per_epoch: Optional[int] = None, verbose: bool = True):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self.history_records: List[Dict[str, Any]] = []
        self.first_bad_batch: Optional[Dict[str, Any]] = None
        self.first_bad_tensor: Optional[str] = None

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        b_idx = batch + 1

        loss_val = logs.get("loss")
        loss_finite = loss_val is not None and math.isfinite(float(loss_val))

        # 1. Weights Finiteness & Range
        weights_finite = True
        bad_weight_name = None
        w_min_val, w_max_val = 0.0, 0.0
        all_w_vals = []
        for w in getattr(self.model, "weights", []):
            try:
                w_arr = w.numpy() if hasattr(w, "numpy") else w
                if not np.all(np.isfinite(w_arr)):
                    weights_finite = False
                    if bad_weight_name is None:
                        bad_weight_name = getattr(w, "name", str(w))
                else:
                    all_w_vals.append(w_arr)
            except Exception:
                pass
        if all_w_vals:
            w_min_val = float(min(np.min(arr) for arr in all_w_vals))
            w_max_val = float(max(np.max(arr) for arr in all_w_vals))

        # 2. Optimizer Slot Variables & Loss Scale
        opt_vars_finite = True
        bad_opt_var_name = None
        loss_scale_val = None
        opt_min_val, opt_max_val = 0.0, 0.0
        all_opt_vals = []

        opt = getattr(self.model, "optimizer", None)
        if opt is not None:
            # Loss scale inspection
            if hasattr(opt, "loss_scale"):
                ls = opt.loss_scale
                loss_scale_val = float(ls.numpy()) if hasattr(ls, "numpy") else float(ls)
            elif hasattr(opt, "inner_optimizer") and hasattr(opt.inner_optimizer, "loss_scale"):
                ls = opt.inner_optimizer.loss_scale
                loss_scale_val = float(ls.numpy()) if hasattr(ls, "numpy") else float(ls)
            elif hasattr(opt, "_loss_scale"):
                ls = opt._loss_scale
                loss_scale_val = float(ls.numpy()) if hasattr(ls, "numpy") else float(ls)

            if hasattr(opt, "variables"):
                try:
                    for v in opt.variables():
                        v_arr = v.numpy() if hasattr(v, "numpy") else v
                        if not np.all(np.isfinite(v_arr)):
                            opt_vars_finite = False
                            if bad_opt_var_name is None:
                                bad_opt_var_name = getattr(v, "name", str(v))
                        else:
                            all_opt_vals.append(v_arr)
                except Exception:
                    pass
        if all_opt_vals:
            opt_min_val = float(min(np.min(arr) for arr in all_opt_vals))
            opt_max_val = float(max(np.max(arr) for arr in all_opt_vals))

        # 3. Metric State Variables
        metrics_finite = True
        bad_metric_name = None
        for m in getattr(self.model, "metrics", []):
            for mv in getattr(m, "variables", []):
                try:
                    mv_arr = mv.numpy() if hasattr(mv, "numpy") else mv
                    if not np.all(np.isfinite(mv_arr)):
                        metrics_finite = False
                        if bad_metric_name is None:
                            bad_metric_name = f"{getattr(m, 'name', 'metric')}:{getattr(mv, 'name', str(mv))}"
                except Exception:
                    pass

        # 4. Non-finite keys in logs
        non_finite_log_keys = [
            k for k, v in logs.items()
            if isinstance(v, (int, float, np.number)) and not math.isfinite(float(v))
        ]

        is_clean = (
            loss_finite
            and weights_finite
            and opt_vars_finite
            and metrics_finite
            and len(non_finite_log_keys) == 0
        )

        record = {
            "batch": b_idx,
            "loss": float(loss_val) if loss_val is not None else None,
            "loss_finite": loss_finite,
            "weights_finite": weights_finite,
            "weights_range": [w_min_val, w_max_val],
            "bad_weight": bad_weight_name,
            "opt_vars_finite": opt_vars_finite,
            "opt_vars_range": [opt_min_val, opt_max_val],
            "bad_opt_var": bad_opt_var_name,
            "loss_scale": loss_scale_val,
            "metrics_finite": metrics_finite,
            "bad_metric": bad_metric_name,
            "non_finite_log_keys": non_finite_log_keys,
            "logs": {
                k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                for k, v in logs.items()
            },
        }
        self.history_records.append(record)

        if not is_clean and self.first_bad_batch is None:
            self.first_bad_batch = record
            if not loss_finite:
                self.first_bad_tensor = f"logs['loss'] ({loss_val})"
            elif not weights_finite:
                self.first_bad_tensor = f"model.weights ({bad_weight_name})"
            elif not opt_vars_finite:
                self.first_bad_tensor = f"optimizer.variables ({bad_opt_var_name})"
            elif not metrics_finite:
                self.first_bad_tensor = f"metric.variables ({bad_metric_name})"
            elif non_finite_log_keys:
                self.first_bad_tensor = f"logs[{non_finite_log_keys[0]}] ({logs[non_finite_log_keys[0]]})"

            logger.error("=" * 75)
            logger.error(f"FORENSIC BATCH LOSS TRACKER: FIRST NON-FINITE DETECTED AT BATCH {b_idx:02d}!")
            logger.error(f"  First Bad Step   : Batch {b_idx}")
            logger.error(f"  First Bad Tensor : {self.first_bad_tensor}")
            logger.error(f"  logs['loss']     : {loss_val}")
            logger.error(f"  weights_finite   : {weights_finite} (bad={bad_weight_name})")
            logger.error(f"  opt_vars_finite  : {opt_vars_finite} (bad={bad_opt_var_name})")
            logger.error(f"  loss_scale       : {loss_scale_val}")
            logger.error(f"  metrics_finite   : {metrics_finite} (bad={bad_metric_name})")
            logger.error(f"  non-finite logs  : {non_finite_log_keys}")
            logger.error("=" * 75)

        if self.verbose:
            ls_str = f" | loss_scale={loss_scale_val}" if loss_scale_val is not None else ""
            loss_str = f"{loss_val:.4f}" if loss_finite else str(loss_val)

            total_steps = self.steps_per_epoch
            if total_steps is None and hasattr(self, "params") and isinstance(self.params, dict):
                total_steps = self.params.get("steps")

            steps_str = f"/{total_steps}" if total_steps is not None else ""

            logger.info(
                f"[BATCH {b_idx:02d}{steps_str}] loss={loss_str} | weights_finite={weights_finite} | "
                f"opt_vars_finite={opt_vars_finite}{ls_str} | metrics_clean={len(non_finite_log_keys)==0}"
            )

    def on_test_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        v_idx = batch + 1
        val_loss_val = logs.get("loss")
        if self.verbose:
            v_str = (
                f"{val_loss_val:.4f}"
                if val_loss_val is not None and math.isfinite(float(val_loss_val))
                else str(val_loss_val)
            )
            total_v_steps = self.params.get("steps") if hasattr(self, "params") and isinstance(self.params, dict) else None
            v_steps_str = f"/{total_v_steps}" if total_v_steps is not None else ""
            logger.info(f"[VAL BATCH {v_idx:02d}{v_steps_str}] val_loss={v_str}")


    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        e_idx = epoch + 1
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        t_str = (
            f"{train_loss:.4f}"
            if train_loss is not None and math.isfinite(float(train_loss))
            else str(train_loss)
        )
        v_str = (
            f"{val_loss:.4f}"
            if val_loss is not None and math.isfinite(float(val_loss))
            else str(val_loss)
        )
        logger.info(
            f"[FORENSIC EPOCH {e_idx:02d} END] train_loss={t_str} | val_loss={v_str}"
        )


def build_callbacks(
    checkpoint_filepath: str,
    tensorboard_dir: str,
    csv_log_path: str,
    monitor_metric: str = "val_pr_auc",
    mode: str = "max",
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 3,
) -> list:
    """Construct full suite of Keras training callbacks including NaN guards."""
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)

    callbacks = [
        BatchLossTrackerCallback(),
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


def inspect_10_batch_losses(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Optional[Dict[int, float]] = None,
    strategy: Optional[tf.distribute.Strategy] = None,
) -> Dict[str, Any]:
    """Inspect every batch in the 10-batch train and 3-batch val sets to trace non-finite loss origins.

    Traces:
    - compiled Keras loss vs. direct float32 BinaryCrossentropy vs. weighted float32 BCE
    - predictions min/max, dtypes, unique targets
    - sample_weight / class_weight application
    """
    logger.info("=" * 75)
    logger.info("10-BATCH STEP-BY-STEP LOSS DIAGNOSTIC AUDIT")
    logger.info("=" * 75)

    if strategy is None:
        strategy = tf.distribute.get_strategy()

    c0 = float(class_weights.get(0, 1.0)) if class_weights else 1.0
    c1 = float(class_weights.get(1, 1.0)) if class_weights else 1.0

    bce_fn = keras.losses.BinaryCrossentropy(from_logits=False)
    bce_none_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    first_bad_train_batch = None
    first_bad_val_batch = None

    # Inspect 10 Training Batches
    train_iter = iter(train_ds)
    for b_idx in range(1, 11):
        try:
            x_b, y_b = next(train_iter)
        except StopIteration:
            break

        y_pred = model(x_b, training=True)
        y_f32 = tf.cast(y_b, tf.float32)
        y_pred_f32 = tf.cast(y_pred, tf.float32)

        pred_min = float(tf.reduce_min(y_pred_f32))
        pred_max = float(tf.reduce_max(y_pred_f32))
        pred_finite = bool(tf.reduce_all(tf.math.is_finite(y_pred_f32)))

        raw_bce = float(bce_fn(y_f32, y_pred_f32))
        sample_w = tf.where(tf.equal(y_f32, 1.0), c1, c0)
        weighted_bce_vec = bce_none_fn(y_f32, y_pred_f32) * sample_w
        weighted_bce = float(tf.reduce_mean(weighted_bce_vec))

        # Compiled Keras loss evaluation
        try:
            if hasattr(model, "compute_loss"):
                compiled_loss_tensor = model.compute_loss(x=x_b, y=y_f32, y_pred=y_pred_f32, sample_weight=sample_w)
            elif hasattr(model, "compiled_loss") and model.compiled_loss is not None:
                compiled_loss_tensor = model.compiled_loss(y_f32, y_pred_f32, sample_weight=sample_w)
            else:
                compiled_loss_tensor = model.loss(y_f32, y_pred_f32)
            compiled_loss_val = float(compiled_loss_tensor)
        except Exception as e:
            compiled_loss_val = float("nan")

        loss_finite = math.isfinite(compiled_loss_val)

        logger.info(
            f"TRAIN BATCH {b_idx:02d}/10: y_unique={np.unique(y_b.numpy().ravel()).tolist()} | "
            f"pred=[{pred_min:.4f}, {pred_max:.4f}] (finite={pred_finite}) | "
            f"raw_bce={raw_bce:.4f} | weighted_bce={weighted_bce:.4f} | compiled_loss={compiled_loss_val:.4f} | "
            f"finite={loss_finite}"
        )

        if not loss_finite or not pred_finite or not math.isfinite(raw_bce):
            if first_bad_train_batch is None:
                first_bad_train_batch = {
                    "batch_index": b_idx,
                    "y_unique": np.unique(y_b.numpy().ravel()).tolist(),
                    "pred_dtype": str(y_pred.dtype),
                    "pred_min": pred_min,
                    "pred_max": pred_max,
                    "pred_finite": pred_finite,
                    "compiled_loss": compiled_loss_val,
                    "raw_bce": raw_bce,
                    "weighted_bce": weighted_bce,
                    "class_weights": {0: c0, 1: c1},
                    "class_weight_active": class_weights is not None,
                }

    # Inspect 3 Validation Batches
    val_iter = iter(val_ds)
    for v_idx in range(1, 4):
        try:
            xv, yv = next(val_iter)
        except StopIteration:
            break

        y_pred_v = model(xv, training=False)
        yv_f32 = tf.cast(yv, tf.float32)
        y_pred_v_f32 = tf.cast(y_pred_v, tf.float32)

        pred_min_v = float(tf.reduce_min(y_pred_v_f32))
        pred_max_v = float(tf.reduce_max(y_pred_v_f32))
        pred_finite_v = bool(tf.reduce_all(tf.math.is_finite(y_pred_v_f32)))

        raw_bce_v = float(bce_fn(yv_f32, y_pred_v_f32))

        try:
            if hasattr(model, "compute_loss"):
                compiled_loss_v_tensor = model.compute_loss(x=xv, y=yv_f32, y_pred=y_pred_v_f32, sample_weight=None)
            elif hasattr(model, "compiled_loss") and model.compiled_loss is not None:
                compiled_loss_v_tensor = model.compiled_loss(yv_f32, y_pred_v_f32, sample_weight=None)
            else:
                compiled_loss_v_tensor = model.loss(yv_f32, y_pred_v_f32)
            compiled_loss_v_val = float(compiled_loss_v_tensor)
        except Exception as e:
            compiled_loss_v_val = float("nan")

        loss_finite_v = math.isfinite(compiled_loss_v_val)

        logger.info(
            f"VAL BATCH   {v_idx:02d}/03: y_unique={np.unique(yv.numpy().ravel()).tolist()} | "
            f"pred=[{pred_min_v:.4f}, {pred_max_v:.4f}] (finite={pred_finite_v}) | "
            f"raw_bce={raw_bce_v:.4f} | compiled_loss={compiled_loss_v_val:.4f} | "
            f"finite={loss_finite_v}"
        )

        if not loss_finite_v or not pred_finite_v or not math.isfinite(raw_bce_v):
            if first_bad_val_batch is None:
                first_bad_val_batch = {
                    "batch_index": v_idx,
                    "y_unique": np.unique(yv.numpy().ravel()).tolist(),
                    "pred_dtype": str(y_pred_v.dtype),
                    "pred_min": pred_min_v,
                    "pred_max": pred_max_v,
                    "pred_finite": pred_finite_v,
                    "compiled_loss": compiled_loss_v_val,
                    "raw_bce": raw_bce_v,
                    "class_weight_active": False,
                }

    logger.info("=" * 75)
    return {
        "first_bad_train_batch": first_bad_train_batch,
        "first_bad_val_batch": first_bad_val_batch,
    }


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
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
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
        callbacks: Optional pre-constructed list of Keras Callback objects.

    Returns:
        Keras History object.
    """
    root = get_project_root()

    if checkpoint_filepath is None:
        checkpoint_filepath = str(
            get_output_dir("checkpoints") / f"{experiment_name}_best.keras"
        )

    tensorboard_dir = str(get_output_dir("logs") / "tensorboard" / experiment_name)
    csv_log_path = str(
        get_output_dir("metrics") / f"{experiment_name}_history.csv"
    )

    patience_es = 5
    patience_lr = 3
    if config and "training" in config:
        patience_es = config["training"].get("early_stopping_patience", 5)
        patience_lr = config["training"].get("reduce_lr_patience", 3)

    if callbacks is None:
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

    # Verify Checkpoint Persistence (Requirements 3, 4, 5)
    verify_checkpoint_persistence(checkpoint_filepath)

    logger.info("Model training completed successfully.")
    return history


def run_forensic_k_experiments(
    architecture: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Optional[Dict[int, float]] = None,
    strategy: Optional[tf.distribute.Strategy] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute controlled forensic experiments EXP_K1, EXP_K2, and EXP_K3 (10 batches max).

    Goal: Pinpoint exact location of NaN inside Keras distributed fit/optimizer/metric path.

    Variants:
    - EXP_K1: Baseline model.fit with class_weight=class_weights under MirroredStrategy + mixed_float16.
    - EXP_K2: model.fit with explicit sample_weight dataset (no class_weight parameter).
    - EXP_K3: model.fit with explicit float32 loss computation path while global policy is mixed_float16.
    """
    from medvision.models.factory import build_model
    from medvision.utils.metrics import get_model_metrics

    logger.info("#" * 75)
    logger.info("RUNNING FORENSIC EXPERIMENTS (EXP_K1, EXP_K2, EXP_K3) - 10 BATCHES MAX")
    logger.info("#" * 75)

    if strategy is None:
        strategy = tf.distribute.get_strategy()

    c0 = float(class_weights.get(0, 1.0)) if class_weights else 1.0
    c1 = float(class_weights.get(1, 1.0)) if class_weights else 1.0

    results = {}

    # ==================== EXP_K1 ====================
    logger.info("=" * 75)
    logger.info("FORENSIC EXP_K1: model.fit(class_weight=class_weights)")
    logger.info("=" * 75)
    with strategy.scope():
        model_k1 = build_model(
            architecture=architecture,
            input_shape=(224, 224, 3),
            learning_rate=1e-4,
            compile_model=True,
            mixed_precision=True,
            config=config,
            strategy=strategy,
        )
    tracker_k1 = BatchLossTrackerCallback(verbose=True)

    with strategy.scope():
        try:
            hist_k1 = model_k1.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                steps_per_epoch=10,
                validation_steps=3,
                class_weight=class_weights,
                callbacks=[tracker_k1],
                verbose=1,
            )
            k1_history = hist_k1.history
        except Exception as e:
            logger.error(f"EXP_K1 exception during fit: {e}")
            k1_history = {"loss": [float("nan")], "val_loss": [float("nan")]}

    results["EXP_K1"] = {
        "history": k1_history,
        "first_bad_batch": tracker_k1.first_bad_batch,
        "first_bad_tensor": tracker_k1.first_bad_tensor,
        "batch_records": tracker_k1.history_records,
    }

    # ==================== EXP_K2 ====================
    logger.info("=" * 75)
    logger.info("FORENSIC EXP_K2: model.fit(sample_weight=equivalent dataset weights)")
    logger.info("=" * 75)
    with strategy.scope():
        model_k2 = build_model(
            architecture=architecture,
            input_shape=(224, 224, 3),
            learning_rate=1e-4,
            compile_model=True,
            mixed_precision=True,
            config=config,
            strategy=strategy,
        )
    tracker_k2 = BatchLossTrackerCallback(verbose=True)

    def add_sample_weights(x, y):
        y_f32 = tf.cast(y, tf.float32)
        sw = tf.where(tf.equal(y_f32, 1.0), c1, c0)
        return x, y, sw

    train_ds_weighted = train_ds.map(add_sample_weights)

    with strategy.scope():
        try:
            hist_k2 = model_k2.fit(
                train_ds_weighted,
                validation_data=val_ds,
                epochs=1,
                steps_per_epoch=10,
                validation_steps=3,
                callbacks=[tracker_k2],
                verbose=1,
            )
            k2_history = hist_k2.history
        except Exception as e:
            logger.error(f"EXP_K2 exception during fit: {e}")
            k2_history = {"loss": [float("nan")], "val_loss": [float("nan")]}

    results["EXP_K2"] = {
        "history": k2_history,
        "first_bad_batch": tracker_k2.first_bad_batch,
        "first_bad_tensor": tracker_k2.first_bad_tensor,
        "batch_records": tracker_k2.history_records,
    }

    # ==================== EXP_K3 ====================
    logger.info("=" * 75)
    logger.info("FORENSIC EXP_K3: model.fit with explicit float32 loss computation")
    logger.info("=" * 75)
    with strategy.scope():
        model_k3 = build_model(
            architecture=architecture,
            input_shape=(224, 224, 3),
            learning_rate=1e-4,
            compile_model=False,
            mixed_precision=True,
            config=config,
            strategy=strategy,
        )
        optimizer_k3 = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
        loss_fn_k3 = keras.losses.BinaryCrossentropy(dtype="float32")
        metrics_k3 = get_model_metrics()
        model_k3.compile(
            optimizer=optimizer_k3,
            loss=loss_fn_k3,
            metrics=metrics_k3,
        )
    tracker_k3 = BatchLossTrackerCallback(verbose=True)

    with strategy.scope():
        try:
            hist_k3 = model_k3.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                steps_per_epoch=10,
                validation_steps=3,
                class_weight=class_weights,
                callbacks=[tracker_k3],
                verbose=1,
            )
            k3_history = hist_k3.history
        except Exception as e:
            logger.error(f"EXP_K3 exception during fit: {e}")
            k3_history = {"loss": [float("nan")], "val_loss": [float("nan")]}

    results["EXP_K3"] = {
        "history": k3_history,
        "first_bad_batch": tracker_k3.first_bad_batch,
        "first_bad_tensor": tracker_k3.first_bad_tensor,
        "batch_records": tracker_k3.history_records,
    }

    # Forensic Summary Comparison
    logger.info("#" * 75)
    logger.info("FORENSIC EXPERIMENTS SUMMARY COMPARISON")
    logger.info("#" * 75)
    for exp_name, exp_res in results.items():
        bad_batch = exp_res["first_bad_batch"]
        bad_tensor = exp_res["first_bad_tensor"]
        last_train_loss = exp_res["history"].get("loss", [None])[-1]
        last_val_loss = exp_res["history"].get("val_loss", [None])[-1]

        if bad_batch is not None:
            logger.info(
                f"{exp_name:7s}: FAIL -> First non-finite at batch {bad_batch['batch']}, "
                f"tensor: {bad_tensor} | train_loss={last_train_loss} | val_loss={last_val_loss}"
            )
        else:
            t_str = f"{last_train_loss:.4f}" if last_train_loss is not None and math.isfinite(float(last_train_loss)) else str(last_train_loss)
            v_str = f"{last_val_loss:.4f}" if last_val_loss is not None and math.isfinite(float(last_val_loss)) else str(last_val_loss)
            logger.info(
                f"{exp_name:7s}: PASS -> All 10 batches finite | "
                f"train_loss={t_str} | val_loss={v_str}"
            )
    logger.info("#" * 75)

    return results

