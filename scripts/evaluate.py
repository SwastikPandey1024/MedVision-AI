"""CLI script to evaluate trained MedVision-AI models on validation or test datasets."""

import argparse
import sys
from pathlib import Path
import keras
import tensorflow as tf

from medvision.config.settings import load_config, get_project_root
from medvision.data.local_dev_loader import load_dev_subset_datasets
from medvision.evaluation import evaluate_model_performance
from medvision.utils.logger import get_logger

logger = get_logger("medvision.evaluate_script")


def parse_args():
    parser = argparse.ArgumentParser(description="MedVision-AI Model Evaluation CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained .keras model checkpoint file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "dev"],
        help="Dataset split to evaluate ('val', 'test', 'dev').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold (default 0.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation reports and plots.",
    )
    return parser.parse_args()


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
    model = keras.models.load_model(ckpt_path, safe_mode=False)

    logger.info(f"Loading '{args.split}' dataset split...")
    train_ds, val_ds, test_ds, _, _, _ = load_dev_subset_datasets(sample_fraction=0.05, batch_size=32)

    eval_ds = val_ds if args.split == "val" else (test_ds if args.split == "test" else train_ds)

    prefix = f"{ckpt_path.stem}_{args.split}"
    res = evaluate_model_performance(
        model=model,
        dataset=eval_ds,
        output_dir=out_dir,
        prefix=prefix,
        threshold=args.threshold,
    )

    m = res["metrics"]
    print("=" * 60)
    print("MEDVISION-AI MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Checkpoint File     : {ckpt_path.name}")
    print(f"Dataset Split       : {args.split}")
    print(f"Total Samples       : {m['sample_count']}")
    print(f"PR-AUC (Primary)    : {m['pr_auc']}")
    print(f"ROC-AUC             : {m['roc_auc']}")
    print(f"F1-Score            : {m['f1_score']}")
    print(f"Recall / Sensitivity: {m['recall_sensitivity']}")
    print(f"Specificity         : {m['specificity']}")
    print(f"Precision           : {m['precision']}")
    print(f"Accuracy            : {m['accuracy']}")
    print(f"Confusion Matrix    : TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}")
    print(f"Report JSON         : {res['json_path']}")
    print(f"Report Markdown     : {res['md_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
