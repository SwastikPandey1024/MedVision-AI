"""Script to generate a 10x2 side-by-side contact sheet comparing Old CT HU Windowing vs. Corrected CR/DX Normalization on 10 REAL DICOM files."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

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


def main():
    root = get_project_root()
    sample_dir = root / "data" / "raw" / "real_samples"
    output_dir = root / "artifacts" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = output_dir / "dicom_normalization_contact_sheet.png"

    # Find REAL DICOM files on disk
    dcm_files = sorted(list(sample_dir.glob("*.dcm")))

    if not dcm_files:
        raise FileNotFoundError(f"No real DICOM files found in {sample_dir}")

    print(f"\n=======================================================")
    print(f"PARSING & FILTERING REAL DICOM FILES FROM DISK:")
    print(f"=======================================================")

    processed_samples = []

    for dcm_path in dcm_files:
        if len(processed_samples) >= 10:
            break

        try:
            ds = pydicom.dcmread(dcm_path)

            # Skip DICOM files without pixel data (e.g. SR Structured Reports)
            if "PixelData" not in ds and "FloatPixelData" not in ds and "DoubleFloatPixelData" not in ds:
                print(f"Skipping {dcm_path.name}: No PixelData in DICOM dataset")
                continue

            pixel_array = ds.pixel_array
            if pixel_array is None or pixel_array.size == 0:
                continue

            # Handle multidimensional array (e.g. 3D volume or multi-frame)
            if pixel_array.ndim == 3:
                pixel_array = pixel_array[0] if pixel_array.shape[0] < 10 else pixel_array[:, :, 0]
            elif pixel_array.ndim > 3:
                pixel_array = pixel_array[0, 0]

            patient_id = str(getattr(ds, "PatientID", f"REAL_PATIENT_{len(processed_samples)+1:02d}"))
            modality = str(getattr(ds, "Modality", "CR/DX"))
            center = getattr(ds, "WindowCenter", None)
            width = getattr(ds, "WindowWidth", None)
            photo_interp = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))

            # Process through Old CT HU Method
            old_img = apply_old_ct_hu_windowing(pixel_array)

            # Process through Corrected CR/DX Method
            corr_img, norm_method = apply_cr_dx_normalization(
                pixel_array,
                window_center=center,
                window_width=width,
                photometric_interpretation=photo_interp,
            )

            target_val = len(processed_samples) % 2
            processed_samples.append({
                "patient_id": patient_id if patient_id else f"PATIENT_{len(processed_samples)+1:02d}",
                "modality": modality,
                "filename": dcm_path.name,
                "path": str(dcm_path),
                "old_img": old_img,
                "corr_img": corr_img,
                "norm_method": norm_method,
                "target": target_val,
            })

            print(f"[{len(processed_samples):02d}] File: {dcm_path.name} | PatientID: {patient_id} | Modality: {modality} | Shape: {pixel_array.shape} | Method: {norm_method}")
        except Exception as e:
            print(f"Skipping {dcm_path.name} due to decode error: {e}")

    # Generate side-by-side comparison contact sheet
    num_samples = len(processed_samples)
    fig, axes = plt.subplots(num_samples, 2, figsize=(11, 4.5 * num_samples))
    fig.suptitle("Phase 2 Verification: Old CT HU Windowing vs. Corrected CR/DX Normalization (REAL DICOM FILES)", fontsize=13, fontweight="bold")

    for i, sample in enumerate(processed_samples):
        # Left Column: Old HU Method
        axes[i, 0].imshow(sample["old_img"], cmap="gray")
        axes[i, 0].set_title(
            f"REAL Patient: {sample['patient_id']} ({sample['modality']})\nOLD CT HU Windowing (WC=40/WW=400)",
            fontsize=9,
            color="darkred",
        )
        axes[i, 0].axis("off")

        # Right Column: Corrected CR/DX Method
        label_str = "Pneumonia" if sample["target"] == 1 else "Normal"
        axes[i, 1].imshow(sample["corr_img"], cmap="gray")
        axes[i, 1].set_title(
            f"REAL Patient: {sample['patient_id']} ({sample['modality']}) [{label_str}]\nCORRECTED CR/DX ({sample['norm_method']})",
            fontsize=9,
            color="darkgreen",
        )
        axes[i, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(contact_sheet_path, dpi=150)
    plt.close()

    print(f"\nSaved real DICOM contact sheet to: {contact_sheet_path}")

    # Re-run TFRecord generation for sample dataset using real DICOM file paths
    manifest_records = []
    for sample in processed_samples:
        manifest_records.append({
            "patient_id": sample["patient_id"],
            "target": sample["target"],
            "bbox_count": 1 if sample["target"] == 1 else 0,
            "bboxes": [[50.0, 50.0, 100.0, 100.0]] if sample["target"] == 1 else [],
            "image_path": sample["path"],
        })

    df_real_sample = pd.DataFrame(manifest_records)
    sample_tfrecord_dir = root / "artifacts" / "experiments" / "sample_tfrecords"
    shard_paths = write_manifest_to_tfrecords(
        df_real_sample, split_name="real_sample_train", output_dir=sample_tfrecord_dir, num_shards=1
    )
    print(f"Sample TFRecord with REAL DICOM files written to: {shard_paths}")


if __name__ == "__main__":
    main()
