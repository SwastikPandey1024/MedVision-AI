"""CLI utility to generate model architecture summary TXT and SVG/PNG diagrams."""

import argparse
import sys
from pathlib import Path
import os

from medvision.config.settings import load_config, get_project_root
from medvision.utils.visualization import visualize_architecture
from medvision.utils.logger import get_logger

logger = get_logger("medvision.visualize_script")


def parse_args():
    parser = argparse.ArgumentParser(description="MedVision-AI Model Architecture Visualization CLI")
    parser.add_argument(
        "--architecture",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "densenet121", "efficientnetb0", "all"],
        help="Architecture to visualize ('custom_cnn', 'densenet121', 'efficientnetb0', 'all').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory (defaults to 'artifacts/architecture/').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    root = get_project_root()
    out_dir = Path(args.output_dir) if args.output_dir else root / "artifacts" / "architecture"
    out_dir.mkdir(parents=True, exist_ok=True)

    archs = ["custom_cnn", "densenet121", "efficientnetb0"] if args.architecture == "all" else [args.architecture]

    print("=" * 70)
    print("MEDVISION-AI MODEL ARCHITECTURE VISUALIZATION GENERATOR")
    print("=" * 70)
    print(f"Target Architecture(s) : {archs}")
    print(f"Output Directory       : {out_dir}")
    print("=" * 70)
    print()

    for arch in archs:
        res = visualize_architecture(architecture=arch, output_dir=out_dir, config=config)
        print(f"Architecture         : {res['architecture']}")
        print(f"Model Identifier     : {res['model_name']}")
        print(f"Input Shape          : {res['input_shape']}")
        print(f"Output Shape         : {res['output_shape']}")
        print(f"Total Parameters     : {res['total_params']:,}")
        print(f"Trainable Parameters : {res['trainable_params']:,}")
        print(f"Non-Trainable Params : {res['non_trainable_params']:,}")
        print(f"Summary TXT Report   : {res['summary_txt_path']}")
        print(f"SVG Diagram          : {res['svg_path']}")
        print(f"PNG Diagram          : {res['png_path']}")
        print("-" * 70)
        print()

    print("Visualization generation completed successfully.")


if __name__ == "__main__":
    main()
