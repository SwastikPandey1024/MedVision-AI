"""Training script for MedVision-AI models (Local CPU Dev-subset or Kaggle Multi-GPU).

CTO-APPROVED CLOUD TRAINING PIPELINE WITH HARD CARDINALITY AND NAN GUARDS:
Phase A: Pre-Flight Safety & Dataset Provenance Checks
Phase B: Data Pipeline Validation (Finite Cardinality & 70/15/15 Patient-Level Split)
Phase C: Training Data Class Weights (Train set ONLY)
Phase D: Single-Batch Real-Data Step Diagnostic (Input -> Pred -> Loss -> Grads -> Opt Update)
Phase D2: 10-Batch Real-Data Performance Benchmark & Finite Cardinality Verification
Phase E: Stage 1 DenseNet121 Head Training (val_pr_auc monitor, TerminateOnNaN + NaNGuardCallback)
Phase F: Best Stage 1 Model Validation Summary & Controlled Stop
"""

import argparse
import sys
import os
import time
import math
import subprocess
from pathlib import Path

# Ensure src/ directory is in sys.path for direct script execution
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

from medvision.config.settings import load_config, get_project_root, get_output_dir
from medvision.data.dataset import find_dataset_root, parse_rsna_manifest, create_real_rsna_dataset
from medvision.data.splits import create_patient_aware_splits
from medvision.data.local_dev_loader import load_dev_subset_datasets
from medvision.data.preprocessing import create_tfrecord_dataset, build_tfrecord_dataset
from medvision.models.factory import build_model, get_distribution_strategy
from medvision.models.densenet import unfreeze_densenet_for_finetuning
from medvision.models.trainer import (
    train_model,
    compute_training_class_weights,
    run_real_batch_diagnostic,
    inspect_10_batch_losses,
    build_callbacks,
    run_forensic_k_experiments,
    verify_checkpoint_persistence,
    find_valid_resume_checkpoint,
)
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
        help="Execution mode: 'dev' for 5 percent local subset, 'full' for complete Kaggle RSNA dataset.",
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
        default=64,
        help="Global batch size across all GPUs (default 64).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="diagnostic",
        choices=[
            "diagnostic",
            "preflight_only",
            "clean_fit_only",
            "stage1_fit",
            "stage1_diagnostic",
            "forensic",
            "exp_k1",
            "exp_k2",
            "exp_k3",
            "exp_a",
            "exp_b",
            "exp_c",
            "stage1",
            "stage2",
            "stage2_fit",
            "all",
        ],
        help="Training stage: 'diagnostic', 'preflight_only', 'clean_fit_only', 'forensic', 'exp_k1', 'exp_k2', 'exp_k3', 'stage1', 'stage2', etc.",
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
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to existing .keras model checkpoint to resume training from.",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Optional safety check: must match the epoch recovered from the checkpoint optimizer state.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the canonical valid Stage 1 checkpoint when one exists.",
    )
    return parser.parse_args()


def check_laptop_safety_and_provenance(mode: str, gpu_count: int) -> bool:
    """Verify hardware, Kaggle environment, and RSNA dataset provenance."""
    is_kaggle = os.path.exists("/kaggle/working") or os.path.exists("/kaggle/input")

    ds_root = find_dataset_root()
    labels_file = ds_root / "stage_2_train_labels.csv"
    if not labels_file.exists():
        labels_file = ds_root / "stage_1_train_labels.csv"
    if not labels_file.exists():
        matches = list(ds_root.glob("*train_labels.csv"))
        if len(matches) > 0:
            labels_file = matches[0]

    real_rsna_dataset_exists = labels_file.exists()

    logger.info(f"REAL_RSNA_DATASET = {'YES' if real_rsna_dataset_exists else 'NO'}")
    if real_rsna_dataset_exists:
        logger.info(f"Dataset Root Resolved: {ds_root}")
        logger.info(f"Labels File Resolved : {labels_file}")

    if mode == "full":
        if not is_kaggle:
            logger.error("=" * 75)
            logger.error("LAPTOP SAFETY GUARD ACTIVATED!")
            logger.error("Full training mode (--mode full) is STRICTLY FORBIDDEN on local laptop.")
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
            kaggle_input = Path("/kaggle/input")
            attached_items = [str(p) for p in kaggle_input.glob("*")] if kaggle_input.exists() else []
            logger.error("=" * 75)
            logger.error("RSNA DATASET PROVENANCE GUARD ACTIVATED!")
            logger.error(f"RSNA dataset labels file not found at '{ds_root}'!")
            logger.error(f"Mounted items under /kaggle/input: {attached_items}")
            logger.error("Synthetic fallback is FORBIDDEN for full mode.")
            logger.error("=" * 75)
            sys.exit(1)

    return real_rsna_dataset_exists


def run_10_batch_benchmark(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    strategy: tf.distribute.Strategy,
    global_batch_size: int,
    num_replicas: int,
    per_replica_batch_size: int,
    expected_train_steps: int,
    expected_val_steps: int,
    class_weights: Optional[dict] = None,
):
    """Phase D2: Measure real-data 10-batch step throughput and verify total finiteness using Keras-native path."""
    logger.info("=" * 75)
    logger.info("REAL RSNA DATASET 10-BATCH PERFORMANCE & FINITENESS BENCHMARK")
    logger.info("=" * 75)

    # Step-by-step 10-batch and 3-val batch compiled loss diagnostic audit
    audit_results = inspect_10_batch_losses(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        strategy=strategy,
    )

    start_time = time.time()
    with strategy.scope():
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            steps_per_epoch=10,
            validation_steps=3,
            class_weight=class_weights,
            verbose=1,
        )

    elapsed = time.time() - start_time
    sec_per_step = elapsed / 10.0

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    train_loss_finite = not (math.isnan(train_loss) or math.isinf(train_loss))
    val_loss_finite = not (math.isnan(val_loss) or math.isinf(val_loss))
    weights_finite = all(bool(tf.reduce_all(tf.math.is_finite(w))) for w in model.weights)

    all_finite = train_loss_finite and val_loss_finite and weights_finite
    est_epoch_min = (sec_per_step * expected_train_steps + (elapsed / 13.0) * expected_val_steps) / 60.0

    has_clipnorm = False
    opt_obj = model.optimizer
    if hasattr(opt_obj, "inner_optimizer"):
        opt_obj = opt_obj.inner_optimizer
    if hasattr(opt_obj, "clipnorm") and opt_obj.clipnorm is not None:
        has_clipnorm = True

    print("\n" + "=" * 70)
    print("FINAL PREFLIGHT VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"TRAIN FIT LOSS                  : {'finite' if train_loss_finite else 'NaN/Inf'} (value={train_loss:.4f})")
    print(f"VALIDATION FIT LOSS             : {'finite' if val_loss_finite else 'NaN/Inf'} (value={val_loss:.4f})")
    print(f"TRAIN PREDICTIONS               : {'finite' if all_finite else 'NaN/Inf'}")
    print(f"VALIDATION PREDICTIONS          : {'finite' if all_finite else 'NaN/Inf'}")
    print(f"GRADIENTS                       : {'finite' if all_finite else 'NaN/Inf'}")
    print(f"WEIGHTS                         : {'finite' if weights_finite else 'NaN/Inf'}")
    print(f"CLIPNORM ACTIVE (1.0)           : {'YES' if has_clipnorm else 'NO'}")
    print(f"BENCHMARK TRAIN STEPS EXECUTED  : 10")
    print(f"BENCHMARK VAL STEPS EXECUTED    : 3")
    print(f"FULL EPOCH PROJECTED TRAIN STEPS: {expected_train_steps}")
    print(f"FULL EPOCH PROJECTED VAL STEPS  : {expected_val_steps}")
    print(f"Global Batch Size               : {global_batch_size}")
    print(f"Replicas                        : {num_replicas}")
    print(f"Per-Replica Batch               : {per_replica_batch_size}")
    print(f"Training Sec/Step               : {sec_per_step:.4f} s")
    print(f"Estimated Epoch Time            : {(sec_per_step * expected_train_steps) / 60.0:.2f} minutes")
    print(f"FINAL PREFLIGHT STATUS          : {'PASS' if all_finite else 'FAIL'}")
    print("=" * 70 + "\n")

    return sec_per_step, (sec_per_step * expected_train_steps) / 60.0, all_finite


def run_isolated_subprocess(args, stage: str) -> bool:
    """Launch an isolated Python subprocess to execute a specific entrypoint stage."""
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        script_path,
        "--mode", args.mode,
        "--stage", stage,
        "--architecture", args.architecture,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
    ]
    if args.mixed_precision:
        cmd.append("--mixed-precision")
    if args.auto_resume:
        cmd.append("--auto-resume")
    if args.resume_from:
        cmd.extend(["--resume-from", args.resume_from])
    if args.resume_epoch is not None:
        cmd.extend(["--resume-epoch", str(args.resume_epoch)])

    logger.info("=" * 75)
    logger.info(f"LAUNCHING ISOLATED SUBPROCESS [STAGE: {stage}]...")
    logger.info(f"Subprocess Command: {' '.join(cmd)}")
    logger.info("=" * 75)

    env = os.environ.copy()
    src_dir = str(get_project_root() / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_dir

    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        logger.error(f"SUBPROCESS [STAGE: {stage}] FAILED with exit code {res.returncode}")
        return False
    logger.info(f"SUBPROCESS [STAGE: {stage}] COMPLETED SUCCESSFULLY.")
    return True


def main():
    args = parse_args()
    root = get_project_root()

    logger.info("=" * 75)
    logger.info("MedVision-AI Controlled Cloud Training Engine")
    logger.info("=" * 75)
    logger.info(f"Execution Mode       : {args.mode}")
    logger.info(f"Model Architecture   : {args.architecture}")
    logger.info(f"Target Stage         : {args.stage}")
    logger.info(f"Max Epochs per Stage : {args.epochs}")
    logger.info(f"Smoke Test Mode      : {args.smoke_test}")

    # TWO-PROCESS ORCHESTRATORS (Run before parent TensorFlow initialization)
    if args.stage == "exp_c":
        logger.info("=" * 75)
        logger.info("EXPERIMENT C: TWO-PROCESS ISOLATED EXECUTION FRAMEWORK")
        logger.info("  PROCESS 1: Preflight Diagnostic & 10-Batch Benchmark Subprocess")
        logger.info("  PROCESS 2: Clean Process model.fit Subprocess (Fresh MirroredStrategy + Model)")
        logger.info("=" * 75)

        p1_ok = run_isolated_subprocess(args, stage="preflight_only")
        if not p1_ok:
            logger.error("EXPERIMENT C FAILED: Process 1 (Preflight) failed.")
            sys.exit(1)

        p2_ok = run_isolated_subprocess(args, stage="clean_fit_only")
        if not p2_ok:
            logger.error("EXPERIMENT C FAILED: Process 2 (Clean Fit) failed.")
            sys.exit(1)

        logger.info("=" * 75)
        logger.info("EXPERIMENT C RESULT: BOTH PROCESSES COMPLETED WITH FINITE LOSS.")
        logger.info("EXPERIMENT C TWO-PROCESS ISOLATION STATUS: PASS")
        logger.info("=" * 75)
        return

    if args.stage == "stage1":
        logger.info("=" * 75)
        logger.info("STAGE 1 TRAINING: TWO-PROCESS ISOLATED EXECUTION FRAMEWORK")
        logger.info("  PROCESS 1: Preflight Diagnostic & 10-Batch Benchmark Subprocess")
        logger.info("  PROCESS 2: Stage 1 Full Head Training Subprocess (5 Epochs)")
        logger.info("=" * 75)

        p1_ok = run_isolated_subprocess(args, stage="preflight_only")
        if not p1_ok:
            logger.error("STAGE 1 ABORTED: Preflight subprocess failed.")
            sys.exit(1)

        p2_ok = run_isolated_subprocess(args, stage="stage1_fit")
        if not p2_ok:
            logger.error("STAGE 1 ABORTED: Stage 1 training subprocess failed.")
            sys.exit(1)

        logger.info("STAGE 1 TRAINING COMPLETED SUCCESSFULLY.")
        return

    if args.stage == "stage2":
        logger.info("=" * 75)
        logger.info("STAGE 2 FINE-TUNING: DEDICATED EXECUTION FRAMEWORK")
        logger.info("  PROCESS 1: Preflight Diagnostic & 10-Batch Benchmark Subprocess")
        logger.info("  PROCESS 2: Stage 2 fine-tuning subprocess starting from validated Stage 1 checkpoint")
        logger.info("=" * 75)

        p1_ok = run_isolated_subprocess(args, stage="preflight_only")
        if not p1_ok:
            logger.error("STAGE 2 ABORTED: Preflight subprocess failed.")
            sys.exit(1)

        p2_ok = run_isolated_subprocess(args, stage="stage2_fit")
        if not p2_ok:
            logger.error("STAGE 2 ABORTED: Stage 2 training subprocess failed.")
            sys.exit(1)

        logger.info("STAGE 2 TRAINING COMPLETED SUCCESSFULLY.")
        return

    config = load_config()

    # Hardware Strategy Setup
    gpus = tf.config.list_physical_devices("GPU")
    gpu_count = len(gpus)
    gpu_names = [g.name for g in gpus] if gpu_count > 0 else ["None (CPU Fallback)"]
    strategy, _ = get_distribution_strategy()
    num_replicas = max(1, strategy.num_replicas_in_sync)

    # Batch Size Semantics (CTO Requirement B)
    global_batch_size = args.batch_size
    assert global_batch_size % num_replicas == 0, (
        f"Global batch size ({global_batch_size}) must be cleanly divisible by "
        f"num_replicas ({num_replicas})."
    )
    per_replica_batch_size = global_batch_size // num_replicas

    # Phase A: Pre-flight safety check
    has_real_rsna = check_laptop_safety_and_provenance(args.mode, gpu_count)

    # Load Dataset (Phase B)
    if args.mode == "dev":
        logger.info("Loading 5% development subset for local CPU verification...")
        sample_frac = config.get("execution", {}).get("sample_fraction_dev", 0.05)
        train_ds, val_ds, test_ds, df_train, df_val, df_test = load_dev_subset_datasets(
            sample_fraction=sample_frac,
            batch_size=per_replica_batch_size,
        )
    else:
        logger.info("Full mode selected: Resolving RSNA dataset root & data pipelines...")
        ds_root = find_dataset_root()
        tfrecord_dir = root / "data" / "processed" / "tfrecords"
        train_shards = list(tfrecord_dir.glob("train_*.tfrecord"))

        if len(train_shards) > 0:
            train_ds = build_tfrecord_dataset(train_shards, batch_size=per_replica_batch_size, is_training=True, repeat=False)
            val_ds = build_tfrecord_dataset(list(tfrecord_dir.glob("val_*.tfrecord")), batch_size=per_replica_batch_size, is_training=False, repeat=False)
            test_ds = build_tfrecord_dataset(list(tfrecord_dir.glob("test_*.tfrecord")), batch_size=per_replica_batch_size, is_training=False, repeat=False)

            manifest_csv = root / "data" / "metadata" / "manifest.csv"
            if manifest_csv.exists():
                df_manifest = pd.read_csv(manifest_csv)
                df_train = df_manifest[df_manifest["split"] == "train"] if "split" in df_manifest.columns else df_manifest
                df_val = df_manifest[df_manifest["split"] == "val"] if "split" in df_manifest.columns else pd.DataFrame()
                df_test = df_manifest[df_manifest["split"] == "test"] if "split" in df_manifest.columns else pd.DataFrame()
            else:
                df_manifest = parse_rsna_manifest(ds_root)
                df_train, df_val, df_test = create_patient_aware_splits(df_manifest)
        else:
            logger.info(f"TFRecord shards not found at '{tfrecord_dir}'. Building real RSNA DICOM datasets directly from '{ds_root}'...")
            df_manifest = parse_rsna_manifest(ds_root)
            df_train, df_val, df_test = create_patient_aware_splits(
                df_manifest, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
            )
            train_ds = create_real_rsna_dataset(df_train, batch_size=per_replica_batch_size, is_training=True)
            val_ds = create_real_rsna_dataset(df_val, batch_size=per_replica_batch_size, is_training=False)
            test_ds = create_real_rsna_dataset(df_test, batch_size=per_replica_batch_size, is_training=False)

    # Compute Explicit Cardinality & Expected Steps (CTO Requirement A)
    n_train = len(df_train) if df_train is not None else 0
    n_val = len(df_val) if df_val is not None else 0
    if args.smoke_test:
        expected_train_steps = 4
        expected_val_steps = 2
    else:
        expected_train_steps = math.ceil(n_train / global_batch_size) if n_train > 0 else 292
        expected_val_steps = math.ceil(n_val / global_batch_size) if n_val > 0 else 63

    logger.info("=" * 70)
    logger.info("BATCH SIZE & CARDINALITY SPECIFICATION")
    logger.info("=" * 70)
    logger.info(f"Global Batch Size         : {global_batch_size}")
    logger.info(f"Replica Count             : {num_replicas}")
    logger.info(f"Per-Replica Batch Size    : {per_replica_batch_size}")
    logger.info(f"Train Samples             : {n_train}")
    logger.info(f"Validation Samples        : {n_val}")
    logger.info(f"Expected Train Steps      : {expected_train_steps}")
    logger.info(f"Expected Validation Steps : {expected_val_steps}")
    logger.info("=" * 70)

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
        logger.info("=" * 60)
        logger.info("Executing Phase 3/4 GPU Smoke Test Validation")
        logger.info("=" * 60)
        policy_name = tf.keras.mixed_precision.global_policy().name
        logger.info(f"Global Batch Size              : {global_batch_size}")
        logger.info(f"Per-Replica Batch Size         : {per_replica_batch_size}")
        logger.info(f"Replicas                       : {num_replicas}")
        logger.info(f"Mixed Precision Policy         : {policy_name}")

        # Run 1-batch step diagnostic for DEV/synthetic dataset before model.fit()
        dev_diag = run_real_batch_diagnostic(
            model=model,
            train_ds=train_ds,
            class_weights=class_weights,
            strategy=strategy,
            is_dev=True,
        )
        if dev_diag["first_failure"] != "NONE (ALL STAGES FINITE)":
            logger.error(f"DEV SMOKE DIAGNOSTIC FAILURE: Non-finite values traced to: {dev_diag['first_failure']}")
            sys.exit(1)

        with strategy.scope():
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                steps_per_epoch=4,
                validation_steps=2,
                class_weight=class_weights,
                verbose=1,
            )

        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]
        assert not math.isnan(train_loss) and not math.isinf(train_loss), "Training loss is NaN/Inf!"
        assert not math.isnan(val_loss) and not math.isinf(val_loss), "Validation loss is NaN/Inf!"
        logger.info(f"Smoke Test Training Loss       : {train_loss:.4f} (finite = True)")
        logger.info(f"Smoke Test Validation Loss     : {val_loss:.4f} (finite = True)")
        logger.info("Phase 3/4 GPU Smoke Test finished SUCCESSFULLY.")
        return

    # FORENSIC STAGE BRANCH: Controlled Forensic Experiments (EXP_K1, EXP_K2, EXP_K3)
    if args.stage in ["forensic", "exp_k1", "exp_k2", "exp_k3"]:
        logger.info("=" * 75)
        logger.info(f"FORENSIC SUITE EXECUTION [STAGE: {args.stage}]")
        logger.info("=" * 75)
        forensic_results = run_forensic_k_experiments(
            architecture=args.architecture,
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            strategy=strategy,
            config=config,
        )
        logger.info("FORENSIC SUITE EXECUTION COMPLETED.")
        return

    # EXPERIMENT A BRANCH: Fresh Model + Fresh Dataset + model.fit ONLY (Clean Baseline)
    if args.stage == "exp_a":
        logger.info("=" * 75)
        logger.info("EXPERIMENT A: CLEAN BASELINE (FRESH MODEL + FRESH DATASET + model.fit ONLY)")
        logger.info("=" * 75)
        exp_a_ckpt = str(get_output_dir("checkpoints") / "exp_a_best.keras")
        callbacks = build_callbacks(
            checkpoint_filepath=exp_a_ckpt,
            tensorboard_dir=str(get_output_dir("logs") / "tensorboard" / "exp_a"),
            csv_log_path=str(get_output_dir("metrics") / "exp_a_history.csv"),
            monitor_metric="val_pr_auc",
            mode="max",
        )
        with strategy.scope():
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                steps_per_epoch=10,
                validation_steps=3,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
            )
        verify_checkpoint_persistence(exp_a_ckpt)
        loss_end = history.history["loss"][-1]
        val_loss_end = history.history["val_loss"][-1]
        logger.info("=" * 75)
        logger.info(f"EXPERIMENT A RESULT: train_loss={loss_end:.4f} | val_loss={val_loss_end:.4f}")
        logger.info(f"EXPERIMENT A STATUS: {'PASS' if math.isfinite(loss_end) and math.isfinite(val_loss_end) else 'FAIL (NaN)'}")
        logger.info("=" * 75)
        return

    # CLEAN FIT ONLY BRANCH (Process 2 of Experiment C)
    if args.stage == "clean_fit_only":
        logger.info("=" * 75)
        logger.info("CLEAN FIT ONLY (PROCESS 2 OF EXP_C: CLEAN PROCESS BOUNDARY)")
        logger.info("=" * 75)
        clean_fit_ckpt = str(get_output_dir("checkpoints") / "clean_fit_best.keras")
        callbacks = build_callbacks(
            checkpoint_filepath=clean_fit_ckpt,
            tensorboard_dir=str(get_output_dir("logs") / "tensorboard" / "clean_fit"),
            csv_log_path=str(get_output_dir("metrics") / "clean_fit_history.csv"),
            monitor_metric="val_pr_auc",
            mode="max",
        )
        with strategy.scope():
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                steps_per_epoch=10,
                validation_steps=3,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
            )
        verify_checkpoint_persistence(clean_fit_ckpt)
        loss_end = history.history["loss"][-1]
        val_loss_end = history.history["val_loss"][-1]
        logger.info("=" * 75)
        logger.info(f"CLEAN FIT RESULT: train_loss={loss_end:.4f} | val_loss={val_loss_end:.4f}")
        logger.info(f"CLEAN FIT STATUS: {'PASS' if math.isfinite(loss_end) and math.isfinite(val_loss_end) else 'FAIL (NaN)'}")
        logger.info("=" * 75)
        if not math.isfinite(loss_end) or not math.isfinite(val_loss_end):
            sys.exit(1)
        return

    # STAGE 1 FIT ONLY BRANCH (Process 2 of Stage 1)
    if args.stage == "stage1_fit":
        logger.info("=" * 75)
        logger.info("STAGE 1 FIT ONLY (PROCESS 2 OF STAGE 1: CLEAN PROCESS BOUNDARY)")
        logger.info("=" * 75)
        stage1_exp_name = f"{args.architecture}_stage1_{args.mode}"
        if args.architecture == "densenet121":
            stage1_ckpt_path = str(get_output_dir("checkpoints") / "densenet121_stage1_best.keras")
        else:
            stage1_ckpt_path = str(get_output_dir("checkpoints") / f"{stage1_exp_name}_best.keras")
        resume_from = args.resume_from
        if args.auto_resume and resume_from is None:
            valid_checkpoint = find_valid_resume_checkpoint(
                stage1_ckpt_path, args.architecture
            )
            resume_from = str(valid_checkpoint.path) if valid_checkpoint else None

        history_s1 = train_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=args.epochs,
            steps_per_epoch=expected_train_steps,
            validation_steps=expected_val_steps,
            class_weights=class_weights,
            checkpoint_filepath=stage1_ckpt_path,
            experiment_name=stage1_exp_name,
            config=config,
            resume_from=resume_from,
            initial_epoch=args.resume_epoch,
            resume_architecture=args.architecture,
        )
        return

    if args.stage == "stage2_fit":
        logger.info("=" * 75)
        logger.info("STAGE 2 FIT ONLY (Clean, standalone fine-tuning from validated Stage 1 checkpoint)")
        logger.info("=" * 75)

        if args.mode != "full":
            raise ValueError("STAGE 2 can only run in --mode full and requires the Kaggle GPU dataset runtime.")

        stage1_ckpt_path = str(get_output_dir("checkpoints") / "densenet121_stage1_best.keras")
        stage2_ckpt_path = str(get_output_dir("checkpoints") / "densenet121_stage2_best.keras")
        stage2_source = resolve_stage2_source_checkpoint(stage1_ckpt_path, stage2_ckpt_path, args.architecture)

        if Path(stage2_ckpt_path).exists() and Path(stage2_ckpt_path).resolve() == stage2_source.path:
            logger.info("STAGE 2 RESUME: resuming from valid Stage 2 checkpoint %s", stage2_source.path)
            model = stage2_source.model
            resume_from = stage2_ckpt_path
            initial_epoch_arg = args.resume_epoch
            stage2_model = model
        else:
            logger.info("STAGE 2 START: loading validated Stage 1 checkpoint %s and unfreezing top 20 layers", stage2_source.path)
            model = stage2_source.model
            stage2_model = unfreeze_densenet_for_finetuning(model, unfreeze_layers=20, learning_rate=1e-5)
            resume_from = None
            initial_epoch_arg = None

        trainable_bn_layers = sum(
            1 for layer in stage2_model.layers if isinstance(layer, keras.layers.BatchNormalization) and layer.trainable
        )
        if trainable_bn_layers != 0:
            raise ValueError(f"STAGE 2 SAFETY VIOLATION! Found {trainable_bn_layers} trainable BatchNorm layers.")

        history_s2 = train_model(
            model=stage2_model,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=args.epochs,
            steps_per_epoch=expected_train_steps,
            validation_steps=expected_val_steps,
            class_weights=class_weights,
            checkpoint_filepath=stage2_ckpt_path,
            experiment_name=f"{args.architecture}_stage2_{args.mode}",
            config=config,
            resume_from=resume_from,
            initial_epoch=initial_epoch_arg,
            resume_architecture=args.architecture,
        )
        verify_checkpoint_persistence(stage2_ckpt_path)
        return

    if args.stage in ["diagnostic", "preflight_only", "stage1_diagnostic"]:
        diag_results = run_real_batch_diagnostic(
            model=model,
            train_ds=train_ds,
            class_weights=class_weights,
            strategy=strategy,
        )

        if diag_results["first_failure"] != "NONE (ALL STAGES FINITE)":
            logger.error(f"CRITICAL DIAGNOSTIC FAILURE: Non-finite values traced to: {diag_results['first_failure']}")
            sys.exit(1)

        sec_per_step, est_epoch_min, is_finite = run_10_batch_benchmark(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            strategy=strategy,
            global_batch_size=global_batch_size,
            num_replicas=num_replicas,
            per_replica_batch_size=per_replica_batch_size,
            expected_train_steps=expected_train_steps,
            expected_val_steps=expected_val_steps,
            class_weights=class_weights,
        )

        if not is_finite:
            logger.error("CRITICAL BENCHMARK FAILURE: Non-finite loss/predictions/gradients detected during 10-batch benchmark!")
            sys.exit(1)

        logger.info("=" * 75)
        logger.info("PREFLIGHT DIAGNOSTIC & BENCHMARK COMPLETED SUCCESSFULLY.")
        logger.info("CONTROLLED STOP ACTIVATED BEFORE STAGE 1 TRAINING.")
        logger.info("=" * 75)
        return

    stage1_exp_name = f"{args.architecture}_stage1_{args.mode}"
    if args.architecture == "densenet121":
        stage1_ckpt_path = str(get_output_dir("checkpoints") / "densenet121_stage1_best.keras")
    else:
        stage1_ckpt_path = str(get_output_dir("checkpoints") / f"{stage1_exp_name}_best.keras")
    resume_from = args.resume_from
    if args.auto_resume and resume_from is None:
        valid_checkpoint = find_valid_resume_checkpoint(stage1_ckpt_path, args.architecture)
        resume_from = str(valid_checkpoint.path) if valid_checkpoint else None

    s1_start_time = time.time()
    history_s1 = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        steps_per_epoch=expected_train_steps,
        validation_steps=expected_val_steps,
        class_weights=class_weights,
        checkpoint_filepath=stage1_ckpt_path,
        experiment_name=stage1_exp_name,
        config=config,
        resume_from=resume_from,
        initial_epoch=args.resume_epoch,
        resume_architecture=args.architecture,
    )
    s1_duration = time.time() - s1_start_time

    # Phase F: Reload Best Stage 1 Model & Compute Validation Metrics
    logger.info(f"Reloading best Stage 1 model from: {stage1_ckpt_path}")
    best_stage1_model = keras.models.load_model(stage1_ckpt_path, safe_mode=False)

    val_y_true_list = []
    val_y_pred_list = []
    for vx, vy in val_ds:
        vp = best_stage1_model.predict(vx, verbose=0)
        val_y_true_list.append(vy.numpy())
        val_y_pred_list.append(vp)

    val_y_true_arr = np.concatenate(val_y_true_list).ravel()
    val_y_pred_arr = np.concatenate(val_y_pred_list).ravel()

    # Optimal threshold from validation set
    th_res = select_optimal_threshold_from_val(val_y_true_arr, val_y_pred_arr, criterion="f1_score")
    opt_threshold = th_res["selected_threshold"]

    val_metrics = compute_classification_metrics(val_y_true_arr, val_y_pred_arr, threshold=opt_threshold)

    # Extract best epoch info
    hist_val_pr_auc = history_s1.history.get("val_pr_auc", [])
    best_epoch_idx = int(np.argmax(hist_val_pr_auc)) + 1 if len(hist_val_pr_auc) > 0 else 1
    best_val_pr_auc = float(np.max(hist_val_pr_auc)) if len(hist_val_pr_auc) > 0 else float(val_metrics["pr_auc"])

    # Check if NaN occurred during training
    nan_occurred = any(
        math.isnan(loss) or math.isinf(loss)
        for loss in history_s1.history.get("loss", []) + history_s1.history.get("val_loss", [])
    )

    print("\n" + "=" * 75)
    print("STAGE 1 MODEL TRAINING & VALIDATION SUMMARY")
    print("=" * 75)
    print(f"Stage 1 Best Epoch       : {best_epoch_idx} / {args.epochs}")
    print(f"Best Val PR-AUC          : {best_val_pr_auc:.4f}")
    print(f"Val ROC-AUC              : {val_metrics['roc_auc']:.4f}")
    print(f"Val Sensitivity / Recall : {val_metrics['recall_sensitivity']:.4f}")
    print(f"Val Specificity          : {val_metrics['specificity']:.4f}")
    print(f"Val F1-Score             : {val_metrics['f1_score']:.4f}")
    print(f"Best Checkpoint Path     : {stage1_ckpt_path}")
    print(f"Stage 1 Duration         : {s1_duration:.2f} s ({s1_duration/60.0:.2f} min)")
    print(f"NaN/Inf Occurred         : {nan_occurred}")
    print("=" * 75 + "\n")

    if args.stage == "stage1":
        logger.info("=" * 75)
        logger.info("STAGE 1 TRAINING COMPLETED SUCCESSFULLY. HARD STOP BEFORE STAGE 2.")
        logger.info("STAGE 2 AND TEST SET EVALUATION WERE NOT EXECUTED.")
        logger.info("=" * 75)
        return

    # Phase G: Stage 2 Controlled Fine-Tuning (Only executed if --stage stage2 or --stage all passed)
    logger.info("=" * 75)
    logger.info("PHASE G: Stage 2 Controlled Fine-Tuning (Top 20 Layers Unfrozen)")
    logger.info("=" * 75)

    with strategy.scope():
        model = unfreeze_densenet_for_finetuning(best_stage1_model, unfreeze_layers=20, learning_rate=1e-5)

    trainable_bn_layers = sum(
        1 for layer in model.layers if isinstance(layer, keras.layers.BatchNormalization) and layer.trainable
    )
    assert trainable_bn_layers == 0, f"CRITICAL SAFETY VIOLATION! Found {trainable_bn_layers} trainable BatchNorm layers!"

    stage2_exp_name = f"{args.architecture}_stage2_{args.mode}"
    stage2_ckpt_path = str(get_output_dir("checkpoints") / f"{stage2_exp_name}_best.keras")

    s2_start_time = time.time()
    history_s2 = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        steps_per_epoch=expected_train_steps,
        validation_steps=expected_val_steps,
        class_weights=class_weights,
        checkpoint_filepath=stage2_ckpt_path,
        experiment_name=stage2_exp_name,
        config=config,
    )
    s2_duration = time.time() - s2_start_time

    logger.info("FULL STAGE 1 AND 2 MODEL TRAINING COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
