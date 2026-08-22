"""Inference and image processing services for MedVision-AI REST API."""

import io
import base64
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

from medvision.utils.model_loader import load_medvision_model
from medvision.explainability.gradcam import (
    auto_detect_target_conv_layer,
    generate_gradcam_explanation,
    overlay_heatmap,
)
from medvision.data.dicom_utils import apply_cr_dx_normalization
from medvision.utils.logger import get_logger

logger = get_logger("medvision.api.services")

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
SUPPORTED_EXTENSIONS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp"}


class ModelService:
    """Singleton service managing the lifecycle and execution of the DenseNet121 model."""

    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.target_layer: Optional[str] = None
        self.threshold: float = 0.60
        self.checkpoint_path: Optional[Path] = None
        self.device: str = "CPU"

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def initialize(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """Load and warm up model if not already loaded."""
        if self.model is not None:
            return

        root = Path(__file__).resolve().parent.parent.parent.parent
        default_ckpt = root / "final_artifacts" / "densenet121_stage2_best.keras"

        target_path = Path(checkpoint_path) if checkpoint_path else default_ckpt
        if not target_path.exists():
            # Fallback check
            alt_ckpt = root / "models" / "checkpoints" / "densenet121_stage2_best.keras"
            if alt_ckpt.exists():
                target_path = alt_ckpt
            else:
                logger.warning(f"Production checkpoint not found at {target_path}. Model loading deferred.")
                return

        self.checkpoint_path = target_path
        self.model = load_medvision_model(target_path)
        self.target_layer = auto_detect_target_conv_layer(self.model)

        # Check compute device
        gpus = tf.config.list_physical_devices("GPU")
        self.device = f"GPU ({len(gpus)} device(s))" if gpus else "CPU"

        # Warm up inference with dummy batch
        dummy_tensor = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        _ = self.model(dummy_tensor, training=False)
        logger.info(f"ModelService initialized successfully on {self.device}. Target layer: {self.target_layer}")

    def is_loaded(self) -> bool:
        return self.model is not None


def array_to_base64_png(arr: np.ndarray) -> str:
    """Encode uint8 RGB numpy array to base64 PNG data string."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def decode_image_bytes(
    file_bytes: bytes,
    filename: str = "",
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Decode raw image bytes into (display_rgb_array, preprocessed_tensor, format_name).

    Args:
        file_bytes: Raw bytes from HTTP upload.
        filename: Original file name for format detection.

    Returns:
        Tuple of (display_rgb uint8 array, preprocessed float32 tensor of shape (1, 224, 224, 3), format string).
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.")

    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    is_dicom = filename.lower().endswith((".dcm", ".dicom")) or file_bytes[:128].find(b"DICM") != -1

    if is_dicom:
        try:
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
            fmt = "DICOM"
        except Exception as e:
            raise ValueError(f"Failed to parse DICOM file: {str(e)}") from e
    else:
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            display_rgb = np.array(pil_img, dtype=np.uint8)
            fmt = pil_img.format if pil_img.format else "Image"
        except Exception as e:
            raise ValueError(f"Failed to decode image payload: {str(e)}") from e

    # Construct preprocessed model input (1, 224, 224, 3) in [0, 1]
    pil_resized = Image.fromarray(display_rgb).resize((224, 224), Image.Resampling.BILINEAR)
    preprocessed = np.array(pil_resized, dtype=np.float32) / 255.0
    preprocessed_tensor = np.expand_dims(preprocessed, axis=0)
    return display_rgb, preprocessed_tensor, fmt
