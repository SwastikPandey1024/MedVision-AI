"""High-performance tf.data.Dataset pipeline and augmentation engine."""

from typing import Tuple, List
import tensorflow as tf


def parse_tfrecord_example(
    serialized_example: tf.Tensor,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Deserialize a single TFRecord example into image float tensor and target label.

    Args:
        serialized_example: Serialized Protocol Buffer tensor.
        target_size: Desired image output shape tuple (height, width).

    Returns:
        Tuple of (image_tensor, target_label).
    """
    feature_description = {
        "patient_id": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "bbox_count": tf.io.FixedLenFeature([], tf.int64),
        "bboxes": tf.io.FixedLenFeature([], tf.string),
        "image_bytes": tf.io.FixedLenFeature([], tf.string),
    }

    features = tf.io.parse_single_example(serialized_example, feature_description)

    # Decode JPEG image
    image = tf.io.decode_jpeg(features["image_bytes"], channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

    target = tf.cast(features["target"], tf.float32)
    target = tf.expand_dims(target, axis=-1)  # Shape (1,)

    return image, target


def apply_augmentations(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply clinically-validated data augmentations per ADR-007 decision.

    Ops:
    - Horizontal Flip (random_flip_left_right)
    - Brightness jitter (max_delta=0.1)
    - Contrast jitter (lower=0.9, upper=1.1)

    Strict Constraints:
    - NO vertical flip (preserves apex-to-base pulmonary orientation).
    - NO extreme shear or rotation (preserves spatial integrity).

    Args:
        image: Image float tensor of shape (H, W, 3) in [0, 1].
        label: Target label tensor.

    Returns:
        Tuple of (augmented_image, label).
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def create_tfrecord_dataset(
    shard_paths: List[str],
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    shuffle_buffer_size: int = 1000,
    drop_remainder: bool = True,
) -> tf.data.Dataset:
    """Build tf.data.Dataset input pipeline from TFRecord shards.

    Args:
        shard_paths: List of TFRecord shard file path strings.
        batch_size: Batch size integer.
        target_size: Target image dimensions.
        is_training: Whether to enable shuffling and augmentations.
        shuffle_buffer_size: Buffer size for shuffling.
        drop_remainder: Whether to drop remainder partial batch for multi-GPU shape consistency.

    Returns:
        Configured tf.data.Dataset object yielding (batch_images, batch_labels).
    """
    dataset = tf.data.TFRecordDataset(shard_paths, num_parallel_reads=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Parse TFRecord examples
    dataset = dataset.map(
        lambda ex: parse_tfrecord_example(ex, target_size=target_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Apply data augmentation if training
    if is_training:
        dataset = dataset.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
