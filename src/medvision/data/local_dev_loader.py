"""Lightweight local development image loader using PIL/OpenCV for dev subset testing."""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from medvision.config.settings import get_project_root
from medvision.data.dicom_utils import read_and_process_dicom
from medvision.data.splits import create_patient_aware_splits
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
        target = float(row.get("target", 0))

        # Decode image using dicom_utils / PIL fallback
        if path and path.endswith(".dcm"):
            img, norm_method = read_and_process_dicom(path, target_size=target_size)
        else:
            img = np.zeros((*target_size, 3), dtype=np.uint8)
            norm_method = "synthetic_fallback"

        images.append(img.astype(np.float32) / 255.0)
        labels.append(target)
        metadata.append({
            "patient_id": row.get("patient_id", "P_UNK"),
            "bbox_count": row.get("bbox_count", 0),
            "norm_method": norm_method,
        })

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.float32).reshape(-1, 1)

    logger.info(f"Local Dev Loader: Loaded batch of {len(images_arr)} samples with shape {images_arr.shape}")
    return images_arr, labels_arr, metadata


def load_dev_subset_datasets(
    sample_fraction: float = 0.05,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load lightweight dev-subset datasets as tf.data.Dataset pipelines for local CPU testing.

    Args:
        sample_fraction: Fraction of full manifest to sample (default 0.05).
        batch_size: Batch size per step.
        target_size: Target image height and width.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, df_train, df_val, df_test).
    """
    root = get_project_root()
    manifest_path = root / "data" / "metadata" / "manifest.csv"

    if manifest_path.exists():
        df_full = pd.read_csv(manifest_path)
        if len(df_full) >= 20:
            num_samples = max(int(len(df_full) * sample_fraction), 20)
            df_dev = df_full.sample(n=num_samples, random_state=42).reset_index(drop=True)
        else:
            logger.info(f"Local manifest has {len(df_full)} rows. Using synthetic dev dataset (100 samples) for split safety.")
            np.random.seed(42)
            n_samples = 100
            targets = np.array([0] * 70 + [1] * 30)
            np.random.shuffle(targets)
            df_dev = pd.DataFrame({
                "patient_id": [f"DEV_PATIENT_{i:03d}" for i in range(n_samples)],
                "target": targets,
                "image_path": [f"data/raw/stage_2_train_images/DEV_PATIENT_{i:03d}.dcm" for i in range(n_samples)],
                "bbox_count": np.random.choice([0, 1, 2], size=n_samples, p=[0.70, 0.25, 0.05]),
            })
    else:
        logger.warning("Manifest CSV not found. Generating synthetic dev dataset for CPU testing.")
        np.random.seed(42)
        n_samples = 100
        targets = np.array([0] * 70 + [1] * 30)
        np.random.shuffle(targets)
        df_dev = pd.DataFrame({
            "patient_id": [f"DEV_PATIENT_{i:03d}" for i in range(n_samples)],
            "target": targets,
            "image_path": [f"data/raw/stage_2_train_images/DEV_PATIENT_{i:03d}.dcm" for i in range(n_samples)],
            "bbox_count": np.random.choice([0, 1, 2], size=n_samples, p=[0.70, 0.25, 0.05]),
        })

    df_train, df_val, df_test = create_patient_aware_splits(
        df_dev, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    def _df_to_tf_dataset(df: pd.DataFrame, is_training: bool = False) -> tf.data.Dataset:
        images_list = []
        labels_list = []

        for _, row in df.iterrows():
            path = str(row.get("image_path", ""))
            target = float(row.get("target", 0))

            if path and path.endswith(".dcm") and (root / path).exists():
                img, _ = read_and_process_dicom(str(root / path), target_size=target_size)
            else:
                # Deterministic synthetic image pattern for local CPU smoke testing
                img = np.random.randint(50, 200, size=(*target_size, 3), dtype=np.uint8)

            images_list.append(img.astype(np.float32) / 255.0)
            labels_list.append(target)

        img_tensor = tf.constant(np.array(images_list, dtype=np.float32))
        lbl_tensor = tf.constant(np.array(labels_list, dtype=np.float32).reshape(-1, 1))

        ds = tf.data.Dataset.from_tensor_slices((img_tensor, lbl_tensor))
        ds = ds.repeat()
        if is_training:
            ds = ds.shuffle(buffer_size=min(len(df), 100))
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _df_to_tf_dataset(df_train, is_training=True)
    val_ds = _df_to_tf_dataset(df_val, is_training=False)
    test_ds = _df_to_tf_dataset(df_test, is_training=False)

    logger.info(
        f"Dev-subset datasets created: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}"
    )

    return train_ds, val_ds, test_ds, df_train, df_val, df_test
