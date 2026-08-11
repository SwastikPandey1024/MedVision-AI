"""Lightweight local development image loader using PIL/OpenCV for dev subset testing."""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
from medvision.data.dicom_utils import read_and_process_dicom
from medvision.utils.logger import get_logger

logger = get_logger("medvision.data.dev_loader")


def load_dev_sample_batch(
    dev_df: pd.DataFrame,
    batch_size: int = 8,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Lightweight loader for CPU local testing on 5% dev-subset manifest only.

    Explicitly NOT expected to be performant; uses standard PIL/OpenCV decoding.

    Args:
        dev_df: Development subset manifest DataFrame.
        batch_size: Sample batch size.
        target_size: Target image dimensions.

    Returns:
        Tuple of (images_array [B, H, W, 3], labels_array [B, 1], metadata_list).
    """
    sample_df = dev_df.head(batch_size)

    images: List[np.ndarray] = []
    labels: List[float] = []
    metadata: List[Dict[str, Any]] = []

    for _, row in sample_df.iterrows():
        path = str(row.get("image_path", ""))
        target = float(row["target"])

        # Decode image using dicom_utils / PIL fallback
        if path and path.endswith(".dcm"):
            img, norm_method = read_and_process_dicom(path, target_size=target_size)
        else:
            img = np.zeros((*target_size, 3), dtype=np.uint8)
            norm_method = "synthetic_fallback"

        images.append(img.astype(np.float32) / 255.0)
        labels.append(target)
        metadata.append({
            "patient_id": row["patient_id"],
            "bbox_count": row.get("bbox_count", 0),
            "norm_method": norm_method,
        })

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.float32).reshape(-1, 1)

    logger.info(f"Local Dev Loader: Loaded batch of {len(images_arr)} samples with shape {images_arr.shape}")
    return images_arr, labels_arr, metadata
