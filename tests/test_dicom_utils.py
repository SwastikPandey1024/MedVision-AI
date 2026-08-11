"""Unit tests for CR/DX radiograph VOI LUT windowing and percentile fallback normalization utility."""

import numpy as np
import pytest
from medvision.data.dicom_utils import apply_cr_dx_normalization, read_and_process_dicom


def test_apply_cr_dx_normalization_tag_voi_lut():
    """Verify tag-based VOI LUT linear windowing."""
    raw_array = np.linspace(0, 4095, 100).astype(np.float32)

    # Window center = 2048, width = 2000
    windowed, method = apply_cr_dx_normalization(
        raw_array, window_center=2048.0, window_width=2000.0, photometric_interpretation="MONOCHROME2"
    )

    assert method == "tag_voi_lut"
    assert windowed.dtype == np.uint8
    assert windowed.min() == 0
    assert windowed.max() == 255


def test_apply_cr_dx_normalization_percentile_fallback():
    """Verify percentile fallback clipping when tags are missing or invalid."""
    raw_array = np.random.randint(500, 3500, size=(100, 100)).astype(np.uint16)

    windowed, method = apply_cr_dx_normalization(
        raw_array, window_center=None, window_width=None, photometric_interpretation="MONOCHROME2"
    )

    assert method == "percentile_fallback"
    assert windowed.dtype == np.uint8
    assert windowed.shape == (100, 100)


def test_monochrome1_inversion():
    """Verify MONOCHROME1 polarity inversion."""
    raw_array = np.array([[0, 4095]], dtype=np.float32)

    win2, _ = apply_cr_dx_normalization(raw_array, window_center=2047.5, window_width=4095, photometric_interpretation="MONOCHROME2")
    win1, _ = apply_cr_dx_normalization(raw_array, window_center=2047.5, window_width=4095, photometric_interpretation="MONOCHROME1")

    assert np.allclose(win1, 255 - win2)


def test_read_and_process_dicom_fallback():
    """Verify fallback handling when file path does not exist."""
    res, method = read_and_process_dicom("non_existent_file.dcm", target_size=(224, 224))
    assert res.shape == (224, 224, 3)
    assert res.dtype == np.uint8
    assert method == "synthetic_fallback"
