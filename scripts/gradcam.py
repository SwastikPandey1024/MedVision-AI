"""CLI script to generate Grad-CAM visual explanations for chest radiographs."""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure src/ is in sys.path
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from PIL import Image
import tensorflow as tf

from medvision.utils.model_loader import load_medvision_model
from medvision.explainability.gradcam import (
    auto_detect_target_conv_layer,
    compute_gradcam_heatmap,
    overlay_heatmap,
    generate_gradcam_explanation,
)
from medvision.data.dicom_utils import read_and_process_dicom
from medvision.utils.logger import get_logger

logger = get_logger("medvision.scripts.gradcam")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MedVision-AI: Generate Grad-CAM Explainability Visualizations."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final_artifacts/densenet121_stage2_best.keras",
        help="Path to trained model checkpoint (.keras file).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file (.dcm, .png, .jpg, .jpeg).",
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default=None,
        help="Path to a directory containing images for batch Grad-CAM generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/explainability",
        help="Directory to save generated visual assets and report.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help="Name of target convolutional layer (auto-detected if omitted).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Decision threshold for pneumonia classification (default: 0.60).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.40,
        help="Overlay blending opacity factor (default: 0.40).",
    )
    return parser.parse_args()


def load_image_for_inference(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load image from disk and return (display_image, preprocessed_tensor)."""
    suffix = file_path.suffix.lower()
    if suffix in [".dcm", ".dicom"]:
        display_img, _ = read_and_process_dicom(str(file_path), target_size=(224, 224))
    else:
        pil_img = Image.open(file_path).convert("RGB")
        display_img = np.array(pil_img)

    # Preprocess for model input (224, 224, 3) in [0, 1]
    pil_resized = Image.fromarray(display_img).resize((224, 224), Image.Resampling.BILINEAR)
    preprocessed = np.array(pil_resized, dtype=np.float32) / 255.0
    preprocessed_tensor = np.expand_dims(preprocessed, axis=0)

    return display_img, preprocessed_tensor


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = load_medvision_model(ckpt_path)

    target_layer = args.target_layer
    if target_layer is None:
        target_layer = auto_detect_target_conv_layer(model)
    logger.info(f"Target convolutional feature layer: '{target_layer}'")

    # Collect image files
    image_paths: List[Path] = []
    if args.image:
        p = Path(args.image)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            image_paths.append(p)
        else:
            logger.error(f"Image not found: {p}")
            sys.exit(1)
    elif args.batch_dir:
        b_dir = Path(args.batch_dir)
        if not b_dir.is_absolute():
            b_dir = root / b_dir
        for ext in ["*.dcm", "*.png", "*.jpg", "*.jpeg"]:
            image_paths.extend(b_dir.glob(ext))
    else:
        # Check for demo test image in root or generate synthetic test sample
        sample_dcm = root / "test_dl.dcm"
        if sample_dcm.exists():
            image_paths.append(sample_dcm)
        else:
            logger.info("No input image specified. Generating synthetic test image for verification.")
            syn_img = out_dir / "synthetic_demo_cxr.png"
            # Create synthetic lung field image
            h, w = 224, 224
            arr = np.zeros((h, w, 3), dtype=np.uint8) + 40
            arr[40:180, 30:95] = 120  # Left lung
            arr[40:180, 125:190] = 120  # Right lung
            arr[90:140, 130:170] = 200  # Focal opacity
            Image.fromarray(arr).save(syn_img)
            image_paths.append(syn_img)

    logger.info(f"Processing {len(image_paths)} images for Grad-CAM generation.")

    results_summary: List[Dict[str, Any]] = []

    for img_path in image_paths:
        try:
            logger.info(f"Generating Grad-CAM for: {img_path.name}")
            display_img, preprocessed_tensor = load_image_for_inference(img_path)

            explanation = generate_gradcam_explanation(
                model=model,
                preprocessed_tensor=preprocessed_tensor,
                original_image=display_img,
                target_layer_name=target_layer,
                threshold=args.threshold,
                alpha=args.alpha,
            )

            stem = img_path.stem
            orig_path = out_dir / f"{stem}_original.png"
            heatmap_path = out_dir / f"{stem}_heatmap.png"
            overlay_path = out_dir / f"{stem}_overlay.png"
            comparison_path = out_dir / f"{stem}_comparison.png"

            # Save visual outputs
            Image.fromarray(explanation["original_image"]).save(orig_path)
            # Save normalized heatmap as colormap
            norm_hm = (explanation["raw_heatmap"] * 255.0).astype(np.uint8)
            Image.fromarray(norm_hm).resize((display_img.shape[1], display_img.shape[0])).save(heatmap_path)
            Image.fromarray(explanation["overlay"]).save(overlay_path)
            Image.fromarray(explanation["side_by_side"]).save(comparison_path)

            record = {
                "file_name": img_path.name,
                "probability": float(explanation["probability"]),
                "prediction": explanation["prediction"],
                "is_pneumonia": explanation["is_pneumonia"],
                "threshold": explanation["threshold"],
                "target_layer": explanation["target_layer"],
                "original_saved": str(orig_path.relative_to(root)),
                "overlay_saved": str(overlay_path.relative_to(root)),
                "comparison_saved": str(comparison_path.relative_to(root)),
            }
            results_summary.append(record)

            print(
                f"[{record['prediction'].upper()}] {img_path.name} | "
                f"Prob: {record['probability']:.4f} (Threshold: {record['threshold']}) | "
                f"Overlay: {overlay_path.name}"
            )

        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")

    # Save summary report
    report_path = out_dir / "gradcam_summary_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path.name),
                "target_layer": target_layer,
                "threshold": args.threshold,
                "total_images_processed": len(results_summary),
                "results": results_summary,
            },
            f,
            indent=2,
        )
    logger.info(f"Grad-CAM summary report written to: {report_path}")


if __name__ == "__main__":
    from typing import Tuple
    main()
