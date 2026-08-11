"""Deterministic seed utility for reproducibility across random, numpy, and tensorflow."""

import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python random, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
