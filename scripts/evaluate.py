"""CLI script to evaluate trained MedVision-AI models on validation or test datasets."""

import argparse
import sys
from pathlib import Path

# Ensure src/ directory is in sys.path for direct script execution
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from typing import Dict, Any, Tuple, Optional, List
import keras
import tensorflow as tf

from medvision.config.settings import load_config, get_project_root

from medvision.data.dataset import find_dataset_root, parse_rsna_manifest, create_real_rsna_dataset
from medvision.data.splits import create_patient_aware_splits
from medvision.data.local_dev_loader import load_dev_subset_datasets
from medvision.data.preprocessing import create_tfrecord_dataset
from medvision.evaluation import (
    evaluate_model_performance,
    generate_model_comparison_report,
    select_optimal_threshold_from_val,
    save_threshold_audit_report,
)
from medvision.utils.metrics import Specificity, F1Score, get_model_metrics
from medvision.utils.logger import get_logger

logger = get_logger("medvision.evaluate_script")



def resolve_evaluation_datasets(
    mode: str = "auto",
    batch_size: int = 32,
    dataset_dir: Optional[Path] = None,
) -> Tuple[str, Dict[str, tf.data.Dataset], Dict[str, Any]]:
    """Resolve and construct evaluation dataset pipelines based on mode and availability.

    Args:
        mode: Resolution mode ('auto', 'full', 'development').
        batch_size: Batch size for evaluation tf.data.Dataset.
        dataset_dir: Optional explicit path to RSNA dataset directory.

    Returns:
        Tuple of (resolved_mode, datasets_dict, metadata_dict).
    """
    root = get_project_root()
    tfrecord_dir = root / "data" / "processed" / "tfrecords"
    val_shards = sorted([str(p) for p in tfrecord_dir.glob("val_*.tfrecord")])
    test_shards = sorted([str(p) for p in tfrecord_dir.glob("test_*.tfrecord")])
    train_shards = sorted([str(p) for p in tfrecord_dir.glob("train_*.tfrecord")])

    has_tfrecords = len(val_shards) > 0 and len(test_shards) > 0

    # Try resolving real RSNA raw dataset path
    has_real_raw = False
    ds_root: Optional[Path] = None
    if dataset_dir is not None:
        candidate_path = Path(dataset_dir)
        if candidate_path.exists() and (
            (candidate_path / "stage_2_train_labels.csv").exists()
            or (candidate_path / "stage_1_train_labels.csv").exists()
            or len(list(candidate_path.glob("*train_labels.csv"))) > 0
        ):
            ds_root = candidate_path
            has_real_raw = True
    else:
        try:
            detected_root = find_dataset_root()
            if detected_root.exists() and (
                (detected_root / "stage_2_train_labels.csv").exists()
                or (detected_root / "stage_1_train_labels.csv").exists()
                or len(list(detected_root.glob("*train_labels.csv"))) > 0
            ):
                ds_root = detected_root
                has_real_raw = True
        except Exception:
            has_real_raw = False

    target_mode = mode.lower()
    if target_mode == "full":
        if not has_tfrecords and not has_real_raw:
            raise FileNotFoundError(
                "Full mode requested, but neither TFRecord shards nor RSNA dataset labels were found."
            )
        resolved_mode = "full"
    elif target_mode == "development":
        resolved_mode = "development"
    elif target_mode == "auto":
        resolved_mode = "full" if (has_tfrecords or has_real_raw) else "development"
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'auto', 'full', 'development'")

    logger.info(f"Resolved evaluation data mode: '{resolved_mode}' (requested: '{mode}')")

    if resolved_mode == "full":
        if has_tfrecords:
            logger.info(f"Loading full evaluation datasets from TFRecord shards in {tfrecord_dir}...")
            val_ds = create_tfrecord_dataset(val_shards, batch_size=batch_size, is_training=False, repeat=False)
            test_ds = create_tfrecord_dataset(test_shards, batch_size=batch_size, is_training=False, repeat=False)
            train_ds = (
                create_tfrecord_dataset(train_shards, batch_size=batch_size, is_training=False, repeat=False)
                if train_shards
                else val_ds
            )
            meta = {
                "source": "tfrecords",
                "val_shards_count": len(val_shards),
                "test_shards_count": len(test_shards),
            }
            return resolved_mode, {"train": train_ds, "val": val_ds, "test": test_ds, "dev": train_ds}, meta
        else:
            logger.info(f"Building official patient-aware 70/15/15 splits from RSNA dataset at {ds_root}...")
            df_manifest = parse_rsna_manifest(ds_root)
            df_train, df_val, df_test = create_patient_aware_splits(
                df_manifest, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
            )
            train_ds = create_real_rsna_dataset(df_train, batch_size=batch_size, is_training=False)
            val_ds = create_real_rsna_dataset(df_val, batch_size=batch_size, is_training=False)
            test_ds = create_real_rsna_dataset(df_test, batch_size=batch_size, is_training=False)
            meta = {
                "source": "real_rsna",
                "n_train": len(df_train),
                "n_val": len(df_val),
                "n_test": len(df_test),
            }
            return resolved_mode, {"train": train_ds, "val": val_ds, "test": test_ds, "dev": train_ds}, meta

    else:
        logger.info("Loading 5% development subset dataset for lightweight evaluation...")
        train_ds, val_ds, test_ds, df_train, df_val, df_test = load_dev_subset_datasets(
            sample_fraction=0.05, batch_size=batch_size
        )
        meta = {
            "source": "dev_subset",
            "n_train": len(df_train),
            "n_val": len(df_val),
            "n_test": len(df_test),
        }
        return resolved_mode, {"train": train_ds, "val": val_ds, "test": test_ds, "dev": train_ds}, meta


def parse_args():
    parser = argparse.ArgumentParser(description="MedVision-AI Model Evaluation CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained .keras model checkpoint file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "full", "development"],
        help="Dataset resolution mode ('auto', 'full', 'development'). Default: 'auto'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "dev", "all"],
        help="Dataset split to evaluate ('val', 'test', 'dev', 'all'). Default: 'val'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for dataset iteration (default: 32).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold (default: 0.5).",
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Optimize decision threshold using validation predictions only before test evaluation.",
    )
    parser.add_argument(
        "--threshold-criterion",
        type=str,
        default="f1_score",
        choices=["f1_score", "accuracy", "youden_j"],
        help="Metric criterion to maximize on validation set during threshold search (default: 'f1_score').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation reports and plots.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional path override for RSNA dataset directory.",
    )
    return parser.parse_args()


def print_evaluation_summary(split_name: str, ckpt_name: str, res: Dict[str, Any], mode: str):
    m = res["metrics"]
    print("=" * 65)
    print(f"MEDVISION-AI MODEL EVALUATION REPORT — [{split_name.upper()}] ({mode.upper()} MODE)")
    print("=" * 65)
    print(f"Checkpoint File     : {ckpt_name}")
    print(f"Dataset Split       : {split_name}")
    print(f"Decision Threshold  : {m['threshold']:.4f}")
    print(f"Total Samples       : {m['sample_count']}")
    print(f"PR-AUC (Primary)    : {m['pr_auc']:.4f}")
    print(f"ROC-AUC             : {m['roc_auc']:.4f}")
    print(f"F1-Score            : {m['f1_score']:.4f}")
    print(f"Recall / Sensitivity: {m['recall_sensitivity']:.4f}")
    print(f"Specificity         : {m['specificity']:.4f}")
    print(f"Precision           : {m['precision']:.4f}")
    print(f"Accuracy            : {m['accuracy']:.4f}")
    print(f"Confusion Matrix    : TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}")
    print(f"Report JSON         : {res['json_path']}")
    print(f"Report Markdown     : {res['md_path']}")
    print("=" * 65)


def main():
    args = parse_args()
    config = load_config()

    root = get_project_root()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path

    if not ckpt_path.exists():
        logger.error(f"Model checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else root / "artifacts" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model checkpoint from: {ckpt_path}")
    model = keras.models.load_model(ckpt_path, compile=False, safe_mode=False)

    resolved_mode, datasets_dict, meta = resolve_evaluation_datasets(
        mode=args.mode,
        batch_size=args.batch_size,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
    )

    evaluated_metrics_list: List[Dict[str, Any]] = []

    if args.optimize_threshold:
        logger.info("=" * 65)
        logger.info("VALIDATION-ONLY DECISION THRESHOLD OPTIMIZATION PHASE")
        logger.info("=" * 65)

        val_ds = datasets_dict.get("val")
        if val_ds is None:
            raise ValueError("Validation dataset required for threshold optimization but not found.")

        # Step A: Evaluate on validation set to collect predictions and ground truth
        val_res = evaluate_model_performance(
            model=model,
            dataset=val_ds,
            output_dir=out_dir,
            prefix=f"{ckpt_path.stem}_val_default0.50",
            threshold=0.5,
        )

        val_y_true = val_res["y_true"]
        val_y_pred_prob = val_res["y_pred_prob"]

        # Step B: Optimize threshold exclusively using validation predictions
        audit_res = select_optimal_threshold_from_val(
            val_y_true=val_y_true,
            val_y_pred_prob=val_y_pred_prob,
            criterion=args.threshold_criterion,
            min_threshold=0.10,
            max_threshold=0.90,
            step=0.01,
        )
        audit_paths = save_threshold_audit_report(
            audit_result=audit_res,
            output_dir=out_dir,
            prefix=f"{ckpt_path.stem}_threshold",
        )

        frozen_threshold = float(audit_res["selected_threshold"])
        logger.info(
            f"FROZEN THRESHOLD SELECTED: {frozen_threshold:.4f} "
            f"(Validation {args.threshold_criterion} = {audit_res['best_val_score']:.4f}). "
            f"Audit saved to {audit_paths['json_path']}"
        )

        # Step C: Evaluate Validation split with optimal threshold
        val_opt_res = evaluate_model_performance(
            model=model,
            dataset=val_ds,
            output_dir=out_dir,
            prefix=f"{ckpt_path.stem}_val",
            threshold=frozen_threshold,
        )
        print_evaluation_summary("val (optimal threshold)", ckpt_path.name, val_opt_res, resolved_mode)

        val_m = dict(val_opt_res["metrics"])
        val_m["model_name"] = f"{ckpt_path.stem} (Val @ th={frozen_threshold:.2f})"
        val_m["params"] = model.count_params()
        evaluated_metrics_list.append(val_m)

        # Step D: Apply FROZEN threshold to Test split (Test data is never seen during optimization)
        if args.split in ["test", "all"]:
            test_ds = datasets_dict.get("test")
            if test_ds is not None:
                test_res = evaluate_model_performance(
                    model=model,
                    dataset=test_ds,
                    output_dir=out_dir,
                    prefix=f"{ckpt_path.stem}_test",
                    threshold=frozen_threshold,
                )
                print_evaluation_summary("test (frozen threshold)", ckpt_path.name, test_res, resolved_mode)

                test_m = dict(test_res["metrics"])
                test_m["model_name"] = f"{ckpt_path.stem} (Held-out Test @ th={frozen_threshold:.2f})"
                test_m["params"] = model.count_params()
                evaluated_metrics_list.append(test_m)

    else:
        splits_to_eval = ["val", "test"] if args.split == "all" else [args.split]

        for split_name in splits_to_eval:
            eval_ds = datasets_dict.get(split_name)
            if eval_ds is None:
                logger.error(f"Requested split '{split_name}' not available in resolved datasets.")
                continue

            prefix = f"{ckpt_path.stem}_{split_name}"
            res = evaluate_model_performance(
                model=model,
                dataset=eval_ds,
                output_dir=out_dir,
                prefix=prefix,
                threshold=args.threshold,
            )

            print_evaluation_summary(split_name, ckpt_path.name, res, resolved_mode)

            split_metric = dict(res["metrics"])
            split_metric["model_name"] = f"{ckpt_path.stem} ({split_name.upper()})"
            split_metric["params"] = model.count_params()
            evaluated_metrics_list.append(split_metric)

    if len(evaluated_metrics_list) > 1:
        comp_res = generate_model_comparison_report(evaluated_metrics_list, output_dir=out_dir)
        logger.info(f"Multi-split comparison report written to: {comp_res['markdown_path']}")


if __name__ == "__main__":
    main()
