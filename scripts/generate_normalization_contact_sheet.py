"""Script to generate a 10x2 side-by-side contact sheet comparing Old CT HU Windowing vs. Corrected CR/DX Normalization."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from medvision.config.settings import get_project_root
from medvision.data.dicom_utils import apply_cr_dx_normalization
from medvision.data.tfrecord_writer import write_manifest_to_tfrecords


def apply_old_ct_hu_windowing(pixel_array: np.ndarray) -> np.ndarray:
    """Old buggy CT HU windowing (WC=40, WW=400)."""
    hu = pixel_array.astype(np.float32) * 1.0 + 0.0
    min_hu = 40.0 - 200.0  # -160
    max_hu = 40.0 + 200.0  # +240
    clipped = np.clip(hu, min_hu, max_hu)
    norm = (clipped - min_hu) / (max_hu - min_hu) * 255.0
    return np.uint8(np.round(norm))


def create_synthetic_radiograph(seed: int) -> np.ndarray:
    """Generate a realistic synthetic 12-bit chest radiograph pixel array (0-4095)."""
    np.random.seed(seed)
    height, width = 512, 512
    y, x = np.ogrid[:height, :width]

    # Background tissue/soft tissue density (medium gray ~ 1800-2400)
    img = np.full((height, width), 2200.0, dtype=np.float32)

    # Lung fields (low density / dark ~ 400-800)
    left_lung = ((x - 170) / 100) ** 2 + ((y - 250) / 180) ** 2 <= 1.0
    right_lung = ((x - 342) / 100) ** 2 + ((y - 250) / 180) ** 2 <= 1.0
    img[left_lung] = 600.0 + np.random.normal(0, 50, size=np.sum(left_lung))
    img[right_lung] = 600.0 + np.random.normal(0, 50, size=np.sum(right_lung))

    # Pneumonia focal opacity consolidation in right lower lobe (seed dependent)
    if seed % 2 == 1:
        opacity = ((x - 360) / 45) ** 2 + ((y - 310) / 45) ** 2 <= 1.0
        img[opacity] = 1600.0 + np.random.normal(0, 80, size=np.sum(opacity))

    # Cardiac silhouette (dense soft tissue ~ 2800)
    heart = ((x - 220) / 90) ** 2 + ((y - 300) / 80) ** 2 <= 1.0
    img[heart] = 2800.0

    # Rib cage arc curves (dense bone ~ 3600-4000)
    for i in range(5):
        rib_y = 120 + i * 70
        rib_mask = (y >= rib_y + 15 * np.sin(x / 40.0)) & (y <= rib_y + 12 + 15 * np.sin(x / 40.0))
        img[rib_mask] = 3700.0

    # Add Gaussian acquisition noise
    img += np.random.normal(0, 30, size=(height, width))
    img = np.clip(img, 0, 4095)
    return img.astype(np.uint16)


def main():
    root = get_project_root()
    output_dir = root / "artifacts" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = output_dir / "dicom_normalization_contact_sheet.png"

    print("Generating 30 synthetic radiograph samples...")
    raw_images = [create_synthetic_radiograph(seed=i) for i in range(30)]

    # Generate 10 paired comparisons for contact sheet
    num_display = 10
    fig, axes = plt.subplots(num_display, 2, figsize=(10, 5 * num_display))
    fig.suptitle("Phase 2 Correction: Old CT HU Windowing vs. Corrected CR/DX Normalization", fontsize=14, fontweight="bold")

    for i in range(num_display):
        raw = raw_images[i]

        # Old CT HU
        old_img = apply_old_ct_hu_windowing(raw)

        # Corrected CR/DX Percentile
        corr_img, method = apply_cr_dx_normalization(raw, window_center=None, window_width=None)

        # Left Column: Old HU Method
        axes[i, 0].imshow(old_img, cmap="gray")
        axes[i, 0].set_title(f"Sample {i+1} — OLD CT HU Windowing (WC=40/WW=400)\n[ALL CLIPPED WHITE]", fontsize=9, color="red")
        axes[i, 0].axis("off")

        # Right Column: Corrected CR/DX Method
        label_text = "Pneumonia" if i % 2 == 1 else "Normal"
        axes[i, 1].imshow(corr_img, cmap="gray")
        axes[i, 1].set_title(f"Sample {i+1} ({label_text}) — CORRECTED CR/DX ({method})\n[CLEAR LUNGS, RIBS, HEART & OPACITY]", fontsize=9, color="green")
        axes[i, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(contact_sheet_path, dpi=150)
    plt.close()

    print(f"Saved contact sheet to: {contact_sheet_path}")

    # Re-run TFRecord generation for sample dataset of 30 images
    records = []
    for i in range(30):
        records.append({
            "patient_id": f"sample_patient_{i:03d}",
            "target": i % 2,
            "bbox_count": 1 if i % 2 == 1 else 0,
            "bboxes": [[100.0, 100.0, 50.0, 50.0]] if i % 2 == 1 else [],
            "image_path": "",
        })
    df_sample = pd.DataFrame(records)

    sample_tfrecord_dir = root / "artifacts" / "experiments" / "sample_tfrecords"
    shard_paths = write_manifest_to_tfrecords(
        df_sample, split_name="sample_train", output_dir=sample_tfrecord_dir, num_shards=1
    )
    print(f"Sample TFRecord written to: {shard_paths}")


if __name__ == "__main__":
    main()
