"""RSNA dataset ingestion, path resolution, and metadata manifest parsing."""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import tensorflow as tf
from medvision.config.settings import get_project_root
from medvision.utils.logger import get_logger

logger = get_logger("medvision.data")


def find_dataset_root() -> Path:
    """Dynamically discover RSNA dataset root path in Kaggle or local environment.

    Returns:
        Path object pointing to existing dataset directory containing stage_2_train_labels.csv or stage_1_train_labels.csv.
    """
    candidate_roots = [
        Path("/kaggle/input/competitions/rsna-pneumonia-detection-challenge"),
        Path("/kaggle/input/rsna-pneumonia-detection-challenge"),
        Path("/kaggle/input/rsna-pneumonia-detection-2018"),
        Path("/kaggle/input/rsna-pneumonia-dataset-in-jpg-format"),
        get_project_root() / "data" / "raw" / "rsna-pneumonia-detection-challenge",
        get_project_root() / "data" / "raw",
    ]

    # 1. Fast direct candidate check (O(1) file existence check)
    for p in candidate_roots:
        if p.exists() and ((p / "stage_2_train_labels.csv").exists() or (p / "stage_1_train_labels.csv").exists()):
            logger.info(f"Dataset root auto-detected at: {p}")
            return p

    # 2. Shallow directory scan over /kaggle/input/competitions and /kaggle/input (NO recursive rglob)
    kaggle_bases = [Path("/kaggle/input/competitions"), Path("/kaggle/input")]
    for base in kaggle_bases:
        if base.exists():
            for p in base.glob("*"):
                if p.is_dir():
                    if (p / "stage_2_train_labels.csv").exists() or (p / "stage_1_train_labels.csv").exists():
                        logger.info(f"Dataset root auto-detected via shallow scan at: {p}")
                        return p

    # 3. Check local raw data directory
    local_raw = get_project_root() / "data" / "raw"
    if local_raw.exists() and ((local_raw / "stage_2_train_labels.csv").exists() or (local_raw / "stage_1_train_labels.csv").exists()):
        logger.info(f"Dataset root auto-detected at local raw directory: {local_raw}")
        return local_raw

    # Fallback to local raw data directory with warning
    logger.warning(f"RSNA dataset labels file not found in standard paths. Defaulting root to: {local_raw}")
    return local_raw


def parse_rsna_manifest(dataset_dir: str | Path | None = None) -> pd.DataFrame:
    """Parse RSNA annotation CSVs and aggregate bounding boxes per patient.

    In the RSNA dataset, positive cases contain multiple rows per patient (one per bbox).
    This function groups by patientId to create a 1-to-1 patient-image manifest.

    Args:
        dataset_dir: Directory containing RSNA dataset files. If None, auto-detected.

    Returns:
        Clean DataFrame with columns: patient_id, target, detailed_class, bbox_count, bboxes, image_path.
    """
    if dataset_dir is None:
        dataset_dir = find_dataset_root()
    else:
        dataset_dir = Path(dataset_dir)

    labels_csv = dataset_dir / "stage_2_train_labels.csv"
    class_info_csv = dataset_dir / "stage_2_detailed_class_info.csv"

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV file not found at: {labels_csv}")

    labels_df = pd.read_csv(labels_csv)

    # Load detailed class info if available
    class_map: Dict[str, str] = {}
    if class_info_csv.exists():
        class_df = pd.read_csv(class_info_csv)
        class_map = dict(zip(class_df["patientId"], class_df["class"]))

    # Group by patientId
    records: List[Dict[str, Any]] = []

    for patient_id, group in labels_df.groupby("patientId"):
        target = int(group["Target"].iloc[0])
        detailed_class = class_map.get(patient_id, "Lung Opacity" if target == 1 else "Normal")

        bboxes: List[List[float]] = []
        if target == 1:
            for _, row in group.iterrows():
                if pd.notna(row["x"]) and pd.notna(row["y"]):
                    bboxes.append([float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])])

        # Resolve image path (.dcm or .jpg/.png fallback)
        dcm_path = dataset_dir / "stage_2_train_images" / f"{patient_id}.dcm"
        png_path = dataset_dir / "stage_2_train_images" / f"{patient_id}.png"
        jpg_path = dataset_dir / "stage_2_train_images" / f"{patient_id}.jpg"

        image_path = str(dcm_path)
        if png_path.exists():
            image_path = str(png_path)
        elif jpg_path.exists():
            image_path = str(jpg_path)

        records.append({
            "patient_id": str(patient_id),
            "target": target,
            "detailed_class": detailed_class,
            "bbox_count": len(bboxes),
            "bboxes": bboxes,
            "image_path": image_path,
        })

    manifest_df = pd.DataFrame(records)
    logger.info(f"Manifest created successfully: {len(manifest_df)} unique patient records parsed.")
    return manifest_df


def create_tf_dataset(
    df: pd.DataFrame,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Build tf.data input pipeline placeholder.

    Args:
        df: Manifest DataFrame.
        image_size: Target image size tuple.
        batch_size: Batch size integer.
        is_training: Training mode flag.

    Returns:
        Configured tf.data.Dataset object.
    """
    raise NotImplementedError("tf.data input pipeline will be implemented in Phase 2.")
