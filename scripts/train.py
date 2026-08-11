"""Training script for MedVision-AI models (Local CPU Dev-subset or Kaggle Multi-GPU)."""

import argparse
import sys
import os
import tensorflow as tf

from medvision.config.settings import load_config, get_project_root
from medvision.data.local_dev_loader import load_dev_subset_datasets
from medvision.models.factory import build_model, get_distribution_strategy
from medvision.models.trainer import train_model, compute_training_class_weights
from medvision.utils.logger import get_logger

logger = get_logger("medvision.train_script")


def parse_args():
    parser = argparse.ArgumentParser(description="MedVision-AI Model Training Script")
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "full"],
        help="Execution mode: 'dev' for 5%% local subset, 'full' for complete dataset.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "densenet121", "efficientnetb0"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (defaults to config settings).",
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


def main():
    args = parse_args()
    config = load_config()
    root = get_project_root()

    logger.info("=" * 60)
    logger.info("MedVision-AI Training Engine Phase 3")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Smoke Test Mode: {args.smoke_test}")

    # Check hardware & distribution strategy
    strategy, gpu_count = get_distribution_strategy()
    logger.info(f"Detected GPUs: {gpu_count} | Distribution Strategy: {strategy.__class__.__name__}")

    # Load Dataset
    if args.mode == "dev":
        logger.info("Loading 5% development subset for local CPU smoke test...")
        sample_frac = config.get("execution", {}).get("sample_fraction_dev", 0.05)
        train_ds, val_ds, test_ds, df_train, df_val, df_test = load_dev_subset_datasets(
            sample_fraction=sample_frac,
            batch_size=args.batch_size,
        )
        epochs = args.epochs if args.epochs is not None else config.get("execution", {}).get("epochs_dev", 2)
    else:
        # Full mode uses TFRecords from Kaggle input path
        logger.info("Full mode selected: Loading TFRecord pipelines...")
        from medvision.data.preprocessing import build_tfrecord_dataset
        tfrecord_dir = root / "data" / "processed" / "tfrecords"
        train_ds = build_tfrecord_dataset(list(tfrecord_dir.glob("train_*.tfrecord")), batch_size=args.batch_size, is_training=True)
        val_ds = build_tfrecord_dataset(list(tfrecord_dir.glob("val_*.tfrecord")), batch_size=args.batch_size, is_training=False)
        test_ds = build_tfrecord_dataset(list(tfrecord_dir.glob("test_*.tfrecord")), batch_size=args.batch_size, is_training=False)

        manifest_csv = root / "data" / "metadata" / "manifest.csv"
        if manifest_csv.exists():
            import pandas as pd
            df_manifest = pd.read_csv(manifest_csv)
            df_train = df_manifest[df_manifest["split"] == "train"] if "split" in df_manifest.columns else df_manifest
        else:
            df_train = None

        epochs = args.epochs if args.epochs is not None else config.get("training", {}).get("epochs_baseline", 15)

    # Compute training class weights using TRAINING DATA ONLY
    if df_train is not None and len(df_train) > 0 and "target" in df_train.columns:
        class_weights = compute_training_class_weights(df_train)
    else:
        logger.warning("Training manifest not found. Class weights set to None.")
        class_weights = None

    # Build model using factory
    model = build_model(
        architecture=args.architecture,
        input_shape=(224, 224, 3),
        learning_rate=config.get("training", {}).get("initial_learning_rate", 1e-4),
        compile_model=True,
        mixed_precision=args.mixed_precision,
        config=config,
    )

    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params

    logger.info(f"Model Summary [{model.name}]:")
    logger.info(f"  Input Shape: {model.input_shape}")
    logger.info(f"  Output Shape: {model.output_shape}")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable Parameters: {non_trainable_params:,}")

    # Short Smoke Test Mode
    if args.smoke_test:
        logger.info("Executing 3-step Smoke Test validation...")
        for step, (x_batch, y_batch) in enumerate(train_ds.take(3)):
            predictions = model(x_batch, training=False)
            loss_val = model.compiled_loss(y_batch, predictions)
            logger.info(
                f"  Smoke Step {step+1}/3 | Batch Shape: {x_batch.shape} | "
                f"Predictions Shape: {predictions.shape} | Loss: {float(loss_val):.4f}"
            )
            assert not tf.math.is_nan(loss_val), "Loss is NaN during smoke test!"
            assert not tf.math.is_inf(loss_val), "Loss is Inf during smoke test!"
        logger.info("Smoke test passed successfully! Output shape and loss are valid.")

        # Test checkpoint writing
        test_ckpt_path = root / "artifacts" / "experiments" / f"{args.architecture}_smoketest.keras"
        model.save(test_ckpt_path)
        logger.info(f"Verified checkpoint creation at: {test_ckpt_path}")
        return

    # Execute full/dev training
    exp_name = f"exp_{args.architecture}_{args.mode}"
    history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        class_weights=class_weights,
        experiment_name=exp_name,
        config=config,
    )

    logger.info("Training script execution finished successfully.")


if __name__ == "__main__":
    main()
