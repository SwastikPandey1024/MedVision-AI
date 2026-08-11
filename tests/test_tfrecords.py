"""Unit tests for TFRecord writing, reading, and round-trip image/metadata serialization."""

import json
import pandas as pd
import pytest
import tensorflow as tf
from medvision.data.preprocessing import parse_tfrecord_example
from medvision.data.tfrecord_writer import create_tf_example, write_manifest_to_tfrecords


def test_tfrecord_example_serialization():
    """Verify single example protocol buffer creation."""
    patient_id = "test_patient_001"
    target = 1
    bbox_count = 1
    bboxes = [[10.0, 20.0, 30.0, 40.0]]
    image_bytes = b"dummy_jpeg_bytes"

    example = create_tf_example(patient_id, target, bbox_count, bboxes, image_bytes)
    serialized = example.SerializeToString()
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


def test_tfrecord_round_trip(tmp_path):
    """Verify manifest records written to TFRecord decode back with matching values."""
    records = [
        {
            "patient_id": "P001",
            "target": 1,
            "bbox_count": 1,
            "bboxes": [[10.0, 20.0, 30.0, 40.0]],
            "image_path": "dummy_1.dcm",
        },
        {
            "patient_id": "P002",
            "target": 0,
            "bbox_count": 0,
            "bboxes": [],
            "image_path": "dummy_2.dcm",
        },
    ]
    df = pd.DataFrame(records)

    shard_paths = write_manifest_to_tfrecords(
        df, split_name="test_split", output_dir=tmp_path, num_shards=1
    )
    assert len(shard_paths) == 1
    assert (tmp_path / "test_split_01-of-01.tfrecord").exists()

    # Read back TFRecord using tf.data
    ds = tf.data.TFRecordDataset(shard_paths)
    parsed_ds = ds.map(lambda ex: parse_tfrecord_example(ex, target_size=(224, 224)))

    batch = list(parsed_ds.take(2))
    assert len(batch) == 2

    img0, target0 = batch[0]
    assert img0.shape == (224, 224, 3)
    assert float(target0.numpy()[0]) == 1.0

    img1, target1 = batch[1]
    assert img1.shape == (224, 224, 3)
    assert float(target1.numpy()[0]) == 0.0
