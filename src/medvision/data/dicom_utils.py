"""DICOM image reading, VOI LUT windowing, and percentile-clipping normalization utility for CR/DX radiographs."""

import os
from typing import Tuple
import numpy as np


def apply_cr_dx_normalization(
    pixel_array: np.ndarray,
    window_center: float | None = None,
    window_width: float | None = None,
    photometric_interpretation: str = "MONOCHROME2",
    percentile_lower: float = 0.5,
    percentile_upper: float = 99.5,
) -> Tuple[np.ndarray, str]:
    """Apply CR/DX modality VOI LUT windowing or percentile fallback clipping.

    CR/DX radiographs represent raw X-ray beam attenuation (no Hounsfield Unit calibration).
    This function applies:
    1. Tag-based VOI LUT linear transform if valid WindowCenter and WindowWidth (>0) exist.
    2. Per-image percentile clipping [0.5%, 99.5%] min-max scaling if tags are absent/invalid.
    3. Photometric Interpretation inversion for MONOCHROME1 (0 = White, Max = Black).

    Args:
        pixel_array: Raw uint16/int16 pixel array from DICOM dataset.
        window_center: DICOM WindowCenter tag value (if present).
        window_width: DICOM WindowWidth tag value (if present).
        photometric_interpretation: DICOM PhotometricInterpretation ('MONOCHROME1' or 'MONOCHROME2').
        percentile_lower: Lower percentile bound for fallback clipping (default 0.5%).
        percentile_upper: Upper percentile bound for fallback clipping (default 99.5%).

    Returns:
        Tuple of (uint8 normalized image array [H, W], method_used string).
    """
    pixels = pixel_array.astype(np.float32)

    # 1. Determine Windowing Path (Tag VOI LUT vs Percentile Fallback)
    use_tag_voi = (
        window_center is not None
        and window_width is not None
        and window_width > 0
    )

    if use_tag_voi:
        wc = float(window_center)
        ww = float(window_width)
        # Standard DICOM VOI LUT linear transformation
        min_val = wc - 0.5 - (ww - 1.0) / 2.0
        max_val = wc - 0.5 + (ww - 1.0) / 2.0
        if max_val > min_val:
            normalized = (pixels - min_val) / (max_val - min_val) * 255.0
        else:
            normalized = np.zeros_like(pixels)
        method_used = "tag_voi_lut"
    else:
        # Fallback to per-image percentile clipping
        p_low = float(np.percentile(pixels, percentile_lower))
        p_high = float(np.percentile(pixels, percentile_upper))

        if p_high > p_low:
            normalized = (pixels - p_low) / (p_high - p_low) * 255.0
        else:
            normalized = np.zeros_like(pixels)
        method_used = "percentile_fallback"

    # Clip to [0, 255] uint8 range
    clipped = np.clip(normalized, 0.0, 255.0)

    # 2. Handle PhotometricInterpretation Polarity
    # MONOCHROME1: 0 is White, Max is Black -> Invert so Bone is White and Air is Black
    if photometric_interpretation.upper() == "MONOCHROME1":
        clipped = 255.0 - clipped

    uint8_img = np.uint8(np.round(clipped))
    return uint8_img, method_used


def read_and_process_dicom(
    file_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, str]:
    """Read DICOM file, apply CR/DX VOI LUT or percentile normalization, and resize.

    Args:
        file_path: Path to DICOM file (.dcm) or image fallback.
        target_size: Output image resolution tuple (height, width).

    Returns:
        Tuple of (uint8 RGB image array [H, W, 3], normalization_method string).
    """
    try:
        import pydicom
        from PIL import Image

        if file_path.endswith(".dcm") and os.path.exists(file_path):
            ds = pydicom.dcmread(file_path)
            pixel_array = ds.pixel_array

            # Extract DICOM WindowCenter / WindowWidth if present
            center = getattr(ds, "WindowCenter", None)
            width = getattr(ds, "WindowWidth", None)

            if isinstance(center, pydicom.multival.MultiValue):
                center = float(center[0])
            elif center is not None:
                center = float(center)

            if isinstance(width, pydicom.multival.MultiValue):
                width = float(width[0])
            elif width is not None:
                width = float(width)

            photo_interp = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))

            norm_img, method_used = apply_cr_dx_normalization(
                pixel_array,
                window_center=center,
                window_width=width,
                photometric_interpretation=photo_interp,
            )

            # Resize using PIL Bilinear interpolation
            img = Image.fromarray(norm_img).resize(target_size, Image.Resampling.BILINEAR)
            img_rgb = img.convert("RGB")
            return np.array(img_rgb, dtype=np.uint8), method_used
    except Exception:
        pass

    # Fallback synthetic array if file not found locally
    dummy = np.zeros((*target_size, 3), dtype=np.uint8)
    return dummy, "synthetic_fallback"
