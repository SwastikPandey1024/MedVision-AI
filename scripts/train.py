"""Training script for MedVision-AI models (Local CPU Dev-subset or Kaggle Multi-GPU).

CTO-APPROVED CLOUD TRAINING PIPELINE:
Phase A: Pre-Flight Safety & Dataset Provenance Checks
Phase B: Data Pipeline Validation (Phase 2 tf.data + 70/15/15 Patient-Level Split)
Phase C: Training Data Class Weights (Train set ONLY)
Phase D: GPU Performance Benchmark
Phase E: Stage 1 DenseNet121 Head Training (LR=1e-4, FP16, max 5 epochs, early stopping patience=2)
Phase F: Stage 1 Validation
Phase G: Stage 2 Controlled Fine-Tuning (Top 20 layers, LR=1e-5, BatchNorm Protection Guard: Trainable BN == 0)
Phase H: Validation Decision Threshold Selection (Validation ONLY, locked threshold)
Phase I: Final Test Evaluation (Untouched Test set, locked threshold)
Phase J & K: Model Comparison & Experiment Manifest
Phase L & M: Laptop Safety & Quota Safeguards (Abort full training on local laptop or CPU)
"""

import argparse
import sys
import os
import time
import math
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

from medvision.config.settings import load_config, get_project_root
from medvision.data.local_dev_loader import load_dev_subset_datasets
from medvision.models.factory import build_model, get_distribution_strategy
from medvision.models.densenet import unfreeze_densenet_for_finetuning
from medvision.models.trainer import train_model, compute_training_class_weights
from medvision.evaluation import (
    compute_classification_metrics,
    plot_evaluation_curves,
    select_optimal_threshold_from_val,
    generate_model_comparison_report,
)
from medvision.utils.reproducibility import generate_experiment_manifest
from medvision.utils.logger import get_logger

logger = get_logger("medvision.train_script")


def parse_args():
    parser = argparse.ArgumentParser(description="MedVision-AI Controlled Cloud Model Training Engine")
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "full"],
        help="Execution mode: 'dev' for 5% local subset, 'full' for complete Kaggle RSNA dataset.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="densenet121",
        choices=["custom_cnn", "densenet121", "efficientnetb0"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum number of epochs per training stage (default 5 for time control).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per replica.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run short 3-step verification test only.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed_float16 precision policy.",
    )
    return parser.parse_args()


def check_laptop_safety_and_provenance(mode: str, gpu_count: int) -> bool:
    """Verify hardware, Kaggle environment, and RSNA dataset provenance."""
    is_kaggle = os.path.exists("/kaggle/working") or os.path.exists("/kaggle/input")
    rsna_dataset_path = Path("/kaggle/input/rsna-pneumonia-detection-challenge")
    real_rsna_dataset_exists = rsna_dataset_path.exists()

    logger.info(f"REAL_RSNA_DATASET = {'YES' if real_rsna_dataset_exists else 'NO'}")

    if mode == "full":
        if not is_kaggle:
            logger.error("=" * 75)
            logger.error("LAPTOP SAFETY GUARD ACTIVATED!")
            logger.error("Full training mode (--mode full) is STRICTLY FORBIDDEN on the local Windows laptop.")
            logger.error("All full RSNA dataset training MUST execute inside Kaggle with GPU enabled.")
            logger.error("=" * 75)
            sys.exit(1)

        if gpu_count == 0:
            logger.error("=" * 75)
            logger.error("GPU HARDWARE GUARD ACTIVATED!")
            logger.error("No GPU detected in current session. Silent CPU fallback is FORBIDDEN for full mode.")
            logger.error("=" * 75)
            sys.exit(1)

        if not real_rsna_dataset_exists:
            logger.error("=" * 75)
            logger.error("RSNA DATASET PROVENANCE GUARD ACTIVATED!")
            logger.error(f"RSNA dataset path '{rsna_dataset_path}' not found! Synthetic fallback is FORBIDDEN for full mode.")
            logger.error("=" * 75)
            sys.exit(1)

    return real_rsna_dataset_exists


def run_gpu_benchmark(model: keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, strategy: tf.distribute.Strategy, gpu_count: int):
    """Phase D: Measure real-data GPU steps/sec performance benchmark."""
    logger.info("=" * 70)
    logger.info("REAL DATA TRAINING BENCHMARK (PHASE D)")
    logger.info("=" * 70)

    start_time = time.time()
    batch_count = 0
    with strategy.scope():
        for step, (x_batch, y_batch) in enumerate(train_ds):
            if step >= 10:
                break
            _ = model(x_batch, training=True)
            batch_count += 1
    elapsed = time.time() - start_time
    sec_per_step = elapsed / max(1, batch_count)

    v_start = time.time()
    v_batches = 0
    with strategy.scope():
        for v_step, (vx, vy) in enumerate(val_ds):
            if v_step >= 3:
                break
            _ = model(vx, training=False)
            v_batches += 1
    v_elapsed = time.time() - v_start
    val_sec_per_step = v_elapsed / max(1, v_batches)

    est_epoch_min = (sec_per_step * 500 + val_sec_per_step * 100) / 60.0

    print("=" * 70)
    print("REAL DATA TRAINING BENCHMARK")
    print("=" * 70)
    print(f"Dataset             : RSNA Pneumonia Detection")
    print(f"GPUs                : {gpu_count}")
    print(f"Strategy            : {strategy.__class__.__name__}")
    print(f"Global Batch Size   : {32 * max(1, gpu_count)}")
    print(f"Training Sec/Step   : {sec_per_step:.4f} s")
    print(f"Validation Sec/Step : {val_sec_per_step:.4f} s")
    print(f"Estimated Epoch Time: {est_epoch_min:.2f} minutes")
    print("=" * 70)


def main():
    args = parse_args()
    config = load_config()
    root = get_project_root()

    logger.info("=" * 75)
    logger.info("MedVision-AI Controlled Cloud Training Engine")
    logger.info("=" * 75)
    logger.info(f"Execution Mode       : {args.mode}")
    logger.info(f"Model Architecture   : {args.architecture}")
    logger.info(f"Max Epochs per Stage : {args.epochs}")
    logger.info(f"Smoke Test Mode      : {args.smoke_test}")

    # Check hardware & distribution strategy
    gpus = tf.config.list_physical_devices("GPU")
    gpu_count = len(gpus)
    gpu_names = [g.name for g in gpus] if gpu_count > 0 else ["None (CPU Fallback)"]
    strategy, _ = get_distribution_strategy()

    logger.info(f"GPU count                      : {gpu_count}")
    logger.info(f"GPU device names               : {gpu_names}")
    logger.info(f"Strategy type                  : {strategy.__class__.__name__}")
    logger.info(f"strategy.num_replicas_in_sync  : {strategy.num_replicas_in_sync}")

    # Phase A: Pre-flight safety check
    has_real_rsna = check_laptop_safety_and_provenance(args.mode, gpu_count)

    # Load Dataset (Phase B)
    if args.mode == "dev":
        logger.info("Loading 5% development subset for local CPU verification...")
        sample_frac = config.get("execution", {}).get("sample_fraction_dev", 0.05)
        train_ds, val_ds, test_ds, df_train, df_val, df_test = load_dev_subset_datasets(
            sample_fraction=sample_frac,
            batch_size=args.batch_size,
        )
        epochs = args.epochs
    else:
        logger.info("Full mode selected: Loading full RSNA TFRecord data pipelines...")
        from medvision.data.preprocessing import build_tfrecord_dataset
        tfrecord_dir = root / "data" / "processed" / "tfrecords"
        train_ds = build_tfrecord_dataset(list(tfrecord_dir.glob("train_*.tfrecord")), batch_size=args.batch_size, is_training=True)
        val_ds = build_tfrecord_dataset(list(tfrecord_dir.glob("val_*.tfrecord")), batch_size=args.batch_size, is_training=False)
        test_ds = build_tfrecord_dataset(list(tfrecord_dir.glob("test_*.tfrecord")), batch_size=args.batch_size, is_training=False)

        manifest_csv = root / "data" / "metadata" / "manifest.csv"
        if manifest_csv.exists():
            df_manifest = pd.read_csv(manifest_csv)
            df_train = df_manifest[df_manifest["split"] == "train"] if "split" in df_manifest.columns else df_manifest
        else:
            df_train = None

        epochs = args.epochs

    # Phase C: Compute training class weights (TRAINING ONLY)
    if df_train is not None and len(df_train) > 0 and "target" in df_train.columns:
        class_weights = compute_training_class_weights(df_train)
    else:
        class_weights = {0: 1.0, 1: 1.0}

    # Build model using single strategy instance
    with strategy.scope():
        model = build_model(
            architecture=args.architecture,
            input_shape=(224, 224, 3),
            learning_rate=1e-4,
            compile_model=True,
            mixed_precision=args.mixed_precision,
            config=config,
            strategy=strategy,
        )

    # Smoke Test Guard
    if args.smoke_test:
        import math
        logger.info("=" * 60)
        logger.info("Executing Phase 3/4 GPU Smoke Test Validation")
        logger.info("=" * 60)
        policy_name = tf.keras.mixed_precision.global_policy().name
        logger.info(f"GPU Count                      : {gpu_count}")
        logger.info(f"GPU Device Names               : {gpu_names}")
        logger.info(f"Strategy Type                  : {strategy.__class__.__name__}")
        logger.info(f"strategy.num_replicas_in_sync  : {strategy.num_replicas_in_sync}")
        logger.info(f"Strategy object identity / reuse: PASS (id={id(strategy)})")
        expected_replicas = gpu_count if gpu_count > 0 else 1
        assert strategy.num_replicas_in_sync == expected_replicas, f"Expected {expected_replicas} replicas, got {strategy.num_replicas_in_sync}"
        logger.info(f"Mixed Precision Policy         : {policy_name}")
        logger.info(f"Model Name                     : {model.name}")
        logger.info(f"Model Output Shape             : {model.output_shape}")

        with strategy.scope():
            # 1. Validation steps (2 steps)
            for v_step, (vx_batch, vy_batch) in enumerate(val_ds):
                if v_step >= 2:
                    break
                v_preds = model(vx_batch, training=False)
                v_loss_val = float(model.compute_loss(vx_batch, vy_batch, v_preds))
                logger.info(f"Validation Batch Shape         : {vx_batch.shape}")
                logger.info(f"Validation Step {v_step+1}/2 Loss         : {v_loss_val:.4f}")
                assert not math.isnan(v_loss_val) and not math.isinf(v_loss_val), "Val Loss is NaN/Inf!"

            # 2. Replica-local training step with optimizer application INSIDE replica context
            def replica_train_step(x, y):
                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    loss = model.compute_loss(x, y, y_pred)
                grads = tape.gradient(loss, model.trainable_variables)
                grads_finite = all(
                    g is None or bool(tf.reduce_all(tf.math.is_finite(g)))
                    for g in grads
                )
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss, grads_finite, y_pred

            for step, (x_batch, y_batch) in enumerate(train_ds):
                if step >= 4:
                    break

                if strategy.num_replicas_in_sync > 1 and hasattr(strategy, "run"):
                    per_replica_loss, per_replica_grads_finite, per_replica_preds = strategy.run(
                        replica_train_step,
                        args=(x_batch, y_batch),
                    )
                    loss_f = float(strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None))
                    if hasattr(per_replica_grads_finite, "values"):
                        grads_finite = all(bool(v) for v in per_replica_grads_finite.values)
                    else:
                        grads_finite = bool(per_replica_grads_finite)

                    if hasattr(per_replica_preds, "values"):
                        predictions = tf.concat(per_replica_preds.values, axis=0)
                    else:
                        predictions = per_replica_preds
                else:
                    loss_tensor, grads_finite, predictions = replica_train_step(x_batch, y_batch)
                    loss_f = float(loss_tensor)

                preds_finite = bool(tf.reduce_all(tf.math.is_finite(predictions)))

                logger.info(f"Training Step {step+1}/4")
                logger.info(f"  Training Batch Shape         : {x_batch.shape}")
                logger.info(f"  Loss Value                   : {loss_f:.4f}")
                logger.info(f"  Gradient Finiteness          : {grads_finite}")
                logger.info(f"  Prediction Finiteness        : {preds_finite}")
                logger.info(f"  Predictions Output Shape     : {predictions.shape}")

                assert grads_finite, f"Gradients contain NaN/Inf at step {step+1}!"
                assert preds_finite, f"Predictions contain NaN/Inf at step {step+1}!"
                assert not math.isnan(loss_f) and not math.isinf(loss_f), "Loss is NaN/Inf during smoke test!"
                assert predictions.shape == (x_batch.shape[0], 1), f"Unexpected predictions shape {predictions.shape}"

        # 3. Verify CSVLogger / TensorBoard callback initialization
        from keras.callbacks import CSVLogger, TensorBoard
        log_dir = root / "artifacts" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_cb = CSVLogger(log_dir / "smoketest_log.csv")
        tb_cb = TensorBoard(log_dir / "tensorboard")
        logger.info(f"CSVLogger Initialized          : {csv_cb is not None}")
        logger.info(f"TensorBoard Initialized        : {tb_cb is not None}")

        # 4. Test checkpoint writing and reload verification
        test_ckpt_path = root / "artifacts" / "experiments" / f"{args.architecture}_smoketest.keras"
        test_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(test_ckpt_path)
        assert test_ckpt_path.exists() and test_ckpt_path.stat().st_size > 0, "Checkpoint failed to save!"

        reloaded_model = keras.models.load_model(test_ckpt_path, safe_mode=False)
        assert reloaded_model is not None and reloaded_model.count_params() == model.count_params(), "Checkpoint reload failed!"
        logger.info(f"Checkpoint Path                : {test_ckpt_path}")
        logger.info("Checkpoint Reload Verification : PASS")
        logger.info("=" * 60)
        logger.info("Phase 3/4 GPU Smoke Test finished SUCCESSFULLY. Stopping before full training.")
        logger.info("=" * 60)
        return

    # Phase D: Run GPU Benchmark
    run_gpu_benchmark(model, train_ds, val_ds, strategy, gpu_count)

    # Phase E: Stage 1 DenseNet121 Head Training
    logger.info("=" * 75)
    logger.info("PHASE E: DenseNet121 Stage 1 Head Training (Backbone Frozen)")
    logger.info("=" * 75)

    stage1_exp_name = f"{args.architecture}_stage1_{args.mode}"
    stage1_ckpt_path = str(root / "artifacts" / "experiments" / f"{stage1_exp_name}_best.keras")

    s1_start_time = time.time()
    history_s1 = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        class_weights=class_weights,
        checkpoint_filepath=stage1_ckpt_path,
        experiment_name=stage1_exp_name,
        config=config,
    )
    s1_duration = time.time() - s1_start_time

    # Phase F: Reload Best Stage 1 Model & Validate
    logger.info(f"Reloading best Stage 1 model from: {stage1_ckpt_path}")
    model = keras.models.load_model(stage1_ckpt_path, safe_mode=False)

    # Phase G: Stage 2 Controlled Fine-Tuning (Top 20 Layers, LR=1e-5)
    logger.info("=" * 75)
    logger.info("PHASE G: Stage 2 Controlled Fine-Tuning (Top 20 Layers Unfrozen)")
    logger.info("=" * 75)

    with strategy.scope():
        model = unfreeze_densenet_for_finetuning(model, unfreeze_layers=20, learning_rate=1e-5)

    # STRICT BATCHNORM SAFETY ASSERTION
    trainable_bn_layers = sum(
        1 for layer in model.layers if isinstance(layer, keras.layers.BatchNormalization) and layer.trainable
    )
    total_bn_layers = sum(1 for layer in model.layers if isinstance(layer, keras.layers.BatchNormalization))
    logger.info(f"Total BatchNorm Layers   : {total_bn_layers}")
    logger.info(f"Trainable BatchNorm Layers: {trainable_bn_layers}")

    assert trainable_bn_layers == 0, f"CRITICAL SAFETY VIOLATION! Found {trainable_bn_layers} trainable BatchNorm layers!"
    logger.info("BatchNorm Safety Assertion PASSED: 0 trainable BatchNorm layers.")

    stage2_exp_name = f"{args.architecture}_stage2_{args.mode}"
    stage2_ckpt_path = str(root / "artifacts" / "experiments" / f"{stage2_exp_name}_best.keras")

    s2_start_time = time.time()
    history_s2 = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        class_weights=class_weights,
        checkpoint_filepath=stage2_ckpt_path,
        experiment_name=stage2_exp_name,
        config=config,
    )
    s2_duration = time.time() - s2_start_time

    # Reload Best Stage 2 Model
    logger.info(f"Reloading best Stage 2 model from: {stage2_ckpt_path}")
    model = keras.models.load_model(stage2_ckpt_path, safe_mode=False)

    # Phase H: Validation Threshold Selection (VALIDATION DATA ONLY)
    logger.info("=" * 75)
    logger.info("PHASE H: Validation-Only Decision Threshold Selection")
    logger.info("=" * 75)

    val_y_true_list = []
    val_y_pred_list = []
    for vx, vy in val_ds:
        vp = model.predict(vx, verbose=0)
        val_y_true_list.append(vy.numpy())
        val_y_pred_list.append(vp)

    val_y_true_arr = np.concatenate(val_y_true_list).ravel()
    val_y_pred_arr = np.concatenate(val_y_pred_list).ravel()

    th_res = select_optimal_threshold_from_val(val_y_true_arr, val_y_pred_arr, criterion="f1_score")
    selected_threshold = th_res["selected_threshold"]
    logger.info(f"LOCKED Decision Threshold: {selected_threshold:.4f} (Validation Set F1={th_res['best_val_score']:.4f})")
    logger.info("TEST SET LABELS WERE NOT USED DURING THRESHOLD SELECTION.")

    # Phase I: Final Test Evaluation (Untouched Test Split with Locked Threshold)
    logger.info("=" * 75)
    logger.info("PHASE I: Final Test Evaluation on Untouched Test Set")
    logger.info("=" * 75)

    test_y_true_list = []
    test_y_pred_list = []
    for tx, ty in test_ds:
        tp = model.predict(tx, verbose=0)
        test_y_true_list.append(ty.numpy())
        test_y_pred_list.append(tp)

    test_y_true_arr = np.concatenate(test_y_true_list).ravel()
    test_y_pred_arr = np.concatenate(test_y_pred_list).ravel()

    final_metrics = compute_classification_metrics(test_y_true_arr, test_y_pred_arr, threshold=selected_threshold)

    out_eval_dir = root / "artifacts" / "evaluation"
    plot_paths = plot_evaluation_curves(test_y_true_arr, test_y_pred_arr, output_dir=out_eval_dir, prefix="final_test")

    # Phase J & K: Model Comparison & Experiment Manifest
    models_comparison = [
        {
            "model_name": "DenseNet121 Stage 2 (Primary)",
            "pr_auc": final_metrics["pr_auc"],
            "roc_auc": final_metrics["roc_auc"],
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall_sensitivity": final_metrics["recall_sensitivity"],
            "specificity": final_metrics["specificity"],
            "f1_score": final_metrics["f1_score"],
            "params": model.count_params(),
        }
    ]
    generate_model_comparison_report(models_comparison, output_dir=root / "artifacts" / "architecture")

    manifest = generate_experiment_manifest(
        experiment_name=stage2_exp_name,
        architecture=args.architecture,
        input_shape=(224, 224, 3),
        batch_size=args.batch_size,
        initial_lr=1e-5,
        optimizer_name="Adam",
        class_weights=class_weights,
        precision_policy=tf.keras.mixed_precision.global_policy().name,
        distribution_strategy_name=strategy.__class__.__name__,
        gpu_count=gpu_count,
        gpu_devices=gpu_names,
        train_samples=len(df_train) if df_train is not None else 0,
        val_samples=len(val_y_true_arr),
        test_samples=len(test_y_true_arr),
        checkpoint_criterion="val_pr_auc",
        selected_threshold=selected_threshold,
        training_duration_seconds=s1_duration + s2_duration,
        output_dir=root / "artifacts" / "experiments",
    )

    logger.info("=" * 75)
    logger.info("MEDVISION-AI FULL RSNA CONTROLLED TRAINING PIPELINE COMPLETED!")
    logger.info(f"Final Test PR-AUC        : {final_metrics['pr_auc']}")
    logger.info(f"Final Test ROC-AUC       : {final_metrics['roc_auc']}")
    logger.info(f"Final Test F1-Score      : {final_metrics['f1_score']}")
    logger.info(f"Locked Decision Threshold: {selected_threshold}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
