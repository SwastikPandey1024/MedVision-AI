"""Streamlit backend inference and explainability caching service."""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import streamlit as st
import keras

from medvision.utils.model_loader import load_medvision_model
from medvision.explainability.gradcam import (
    auto_detect_target_conv_layer,
    generate_gradcam_explanation,
)
from medvision.data.dicom_utils import apply_cr_dx_normalization


@st.cache_resource(show_spinner="Loading DenseNet121 model weights...")
def get_cached_model(checkpoint_path: Optional[str] = None) -> Tuple[keras.Model, str]:
    """Load and cache DenseNet121 model and detect target conv layer."""
    root = Path(__file__).resolve().parent.parent.parent
    if checkpoint_path is None:
        primary_ckpt = root / "final_artifacts" / "densenet121_stage2_best.keras"
        fallback_ckpt = root / "models" / "checkpoints" / "densenet121_stage2_best.keras"
        ckpt_path = primary_ckpt if primary_ckpt.exists() else fallback_ckpt
    else:
        ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

    model = load_medvision_model(ckpt_path)
    target_layer = auto_detect_target_conv_layer(model)
    return model, target_layer


def process_uploaded_image(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Process uploaded file bytes (DICOM or standard image) into display RGB and preprocessed tensor."""
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    is_dicom = filename.lower().endswith((".dcm", ".dicom")) or file_bytes[:128].find(b"DICM") != -1

    if is_dicom:
        import pydicom
        from pydicom.filebase import DicomBytesIO

        ds = pydicom.dcmread(DicomBytesIO(file_bytes))
        pixel_array = ds.pixel_array

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
        norm_uint8, _ = apply_cr_dx_normalization(
            pixel_array,
            window_center=center,
            window_width=width,
            photometric_interpretation=photo_interp,
        )
        display_rgb = np.stack([norm_uint8] * 3, axis=-1)
    else:
        import io
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        display_rgb = np.array(pil_img, dtype=np.uint8)

    # Preprocessed tensor (1, 224, 224, 3) in [0, 1]
    pil_resized = Image.fromarray(display_rgb).resize((224, 224), Image.Resampling.BILINEAR)
    preprocessed = np.array(pil_resized, dtype=np.float32) / 255.0
    preprocessed_tensor = np.expand_dims(preprocessed, axis=0)

    return display_rgb, preprocessed_tensor
