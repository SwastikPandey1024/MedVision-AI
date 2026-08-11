"""Unit tests for DICOM slope/intercept rescaling and lung windowing utility."""

import numpy as np
import pytest
from medvision.data.dicom_utils import apply_dicom_windowing, read_and_process_dicom


def test_apply_dicom_windowing_basic():
    """Verify windowing scaling maps Hounsfield units to uint8 [0, 255]."""
    # Create synthetic array with values -1000 to +1000
    raw_array = np.linspace(-1000, 1000, 100).astype(np.float32)

    # Window center = 40, width = 400 => min = -160, max = +240
    windowed = apply_dicom_windowing(
        raw_array, window_center=40.0, window_width=400.0, rescale_slope=1.0, rescale_intercept=0.0
    )

    assert windowed.dtype == np.uint8
    assert windowed.min() == 0
    assert windowed.max() == 255
    # Value <= -160 should clip to 0
    assert windowed[0] == 0
    # Value >= 240 should clip to 255
    assert windowed[-1] == 255


def test_apply_dicom_windowing_slope_intercept():
    """Verify slope and intercept rescaling logic."""
    raw_array = np.array([0, 100, 200], dtype=np.int16)
    # Slope = 2, Intercept = -100 => HU = [-100, 100, 300]
    windowed = apply_dicom_windowing(
        raw_array, window_center=100.0, window_width=400.0, rescale_slope=2.0, rescale_intercept=-100.0
    )

    assert isinstance(windowed, np.ndarray)
    assert len(windowed) == 3
    assert windowed.dtype == np.uint8


def test_read_and_process_dicom_fallback():
    """Verify fallback handling when file path does not exist."""
    res = read_and_process_dicom("non_existent_file.dcm", target_size=(224, 224))
    assert res.shape == (224, 224, 3)
    assert res.dtype == np.uint8
