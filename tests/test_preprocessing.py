"""Unit tests verifying tf.data.Dataset pipeline shapes, batching, and augmentation toggles."""

import pandas as pd
import pytest
import tensorflow as tf
from medvision.data.local_dev_loader import load_dev_sample_batch
from medvision.data.preprocessing import apply_augmentations, create_tfrecord_dataset
from medvision.data.tfrecord_writer import write_manifest_to_tfrecords


def test_apply_augmentations():
    """Verify data augmentation returns float tensor with shape preservation."""
    dummy_img = tf.random.uniform((224, 224, 3), minval=0.0, maxval=1.0)
    dummy_label = tf.constant([1.0])

    aug_img, aug_label = apply_augmentations(dummy_img, dummy_label)
    assert aug_img.shape == (224, 224, 3)
    assert aug_label.shape == (1,)
    assert aug_img.dtype == tf.float32


def test_tfrecord_dataset_pipeline(tmp_path):
    """Verify create_tfrecord_dataset batches and yields tensors correctly."""
    records = [
        {"patient_id": f"P{i:03d}", "target": i % 2, "bbox_count": 0, "bboxes": [], "image_path": ""}
        for i in range(10)
    ]
    df = pd.DataFrame(records)

    shard_paths = write_manifest_to_tfrecords(
        df, split_name="train", output_dir=tmp_path, num_shards=2
    )

    dataset = create_tfrecord_dataset(
        shard_paths, batch_size=4, target_size=(224, 224), is_training=True
    )

    for images, labels in dataset.take(1):
        assert images.shape == (4, 224, 224, 3)
        assert labels.shape == (4, 1)


def test_local_dev_loader():
    """Verify lightweight local dev loader yields sample batch."""
    records = [
        {"patient_id": f"P{i:03d}", "target": i % 2, "bbox_count": 0, "bboxes": [], "image_path": ""}
        for i in range(5)
    ]
    df = pd.DataFrame(records)

    imgs, labels, meta = load_dev_sample_batch(df, batch_size=3, target_size=(224, 224))
    assert imgs.shape == (3, 224, 224, 3)
    assert labels.shape == (3, 1)
    assert len(meta) == 3
