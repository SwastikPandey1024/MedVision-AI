"""TFRecord serialization and sharding engine for MedVision-AI datasets."""

import io
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from medvision.config.settings import get_project_root
from medvision.data.dicom_utils import read_and_process_dicom
from medvision.utils.logger import get_logger

logger = get_logger("medvision.data.tfrecords")


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / int / enum."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_example(
    patient_id: str,
    target: int,
    bbox_count: int,
    bboxes: List[List[float]],
    image_bytes: bytes,
) -> tf.train.Example:
    """Serialize a single patient sample into a tf.train.Example protocol buffer.

    Args:
        patient_id: Unique patient identifier string.
        target: Classification target integer (0 or 1).
        bbox_count: Number of bounding boxes.
        bboxes: List of bounding box coordinate lists.
        image_bytes: Encoded JPEG image bytes.

    Returns:
        tf.train.Example message object.
    """
    feature = {
        "patient_id": _bytes_feature(patient_id.encode("utf-8")),
        "target": _int64_feature(target),
        "bbox_count": _int64_feature(bbox_count),
        "bboxes": _bytes_feature(json.dumps(bboxes).encode("utf-8")),
        "image_bytes": _bytes_feature(image_bytes),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_manifest_to_tfrecords(
    df: pd.DataFrame,
    split_name: str,
    output_dir: str | Path | None = None,
    target_size: Tuple[int, int] = (224, 224),
    num_shards: int = 4,
) -> List[str]:
    """Convert manifest DataFrame into TFRecord shards.

    Args:
        df: Input manifest DataFrame for a specific split (train/val/test).
        split_name: Name identifier of split ('train', 'val', 'test').
        output_dir: Directory path to save TFRecord files.
        target_size: Target image resolution tuple.
        num_shards: Number of output shard files to create.

    Returns:
        List of generated TFRecord file path strings.
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "processed" / "tfrecords"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    records_per_shard = int(np.ceil(len(df) / num_shards))
    shard_paths: List[str] = []

    for shard_idx in range(num_shards):
        start_idx = shard_idx * records_per_shard
        end_idx = min((shard_idx + 1) * records_per_shard, len(df))
        shard_df = df.iloc[start_idx:end_idx]

        if len(shard_df) == 0:
            continue

        shard_filename = f"{split_name}_{shard_idx+1:02d}-of-{num_shards:02d}.tfrecord"
        shard_path = str(output_dir / shard_filename)
        shard_paths.append(shard_path)

        with tf.io.TFRecordWriter(shard_path) as writer:
            for _, row in shard_df.iterrows():
                patient_id = row["patient_id"]
                target = int(row["target"])
                bbox_count = int(row.get("bbox_count", 0))
                bboxes = row.get("bboxes", [])
                image_path = row.get("image_path", "")

                # Read and process image
                if os.path.exists(image_path) and image_path.endswith(".dcm"):
                    img_array = read_and_process_dicom(image_path, target_size=target_size)
                else:
                    # Synthetic / RGB array fallback for fast dev testing
                    img_array = np.zeros((*target_size, 3), dtype=np.uint8)

                # Encode to JPEG bytes stream
                pil_img = Image.fromarray(img_array)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=95)
                image_bytes = buf.getvalue()

                example = create_tf_example(patient_id, target, bbox_count, bboxes, image_bytes)
                writer.write(example.SerializeToString())

    logger.info(f"Wrote {len(df)} records into {len(shard_paths)} TFRecord shards for split '{split_name}'.")
    return shard_paths
