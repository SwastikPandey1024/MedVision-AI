"""Streamlit backend inference and explainability caching service."""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import streamlit as st
import requests
import keras

from medvision.utils.model_loader import load_medvision_model
from medvision.explainability.gradcam import (
    auto_detect_target_conv_layer,
    generate_gradcam_explanation,
)
from medvision.data.dicom_utils import apply_cr_dx_normalization
from medvision.utils.logger import get_logger

logger = get_logger("medvision.app.inference_service")

# Model filename constants
DEFAULT_CHECKPOINT_FILENAME = "densenet121_stage2_best.keras"


def download_model_checkpoint(
    url: str,
    destination_path: Union[str, Path],
    timeout: int = 120,
    chunk_size: int = 65536,
) -> Path:
    """Download model weights from a remote URL with atomic file writing.

    Args:
        url: Direct downloadable URL to the .keras model artifact.
        destination_path: Target local file path where model will be saved.
        timeout: HTTP request timeout in seconds.
        chunk_size: Stream buffer size in bytes.

    Returns:
        Path to the verified downloaded checkpoint.

    Raises:
        RuntimeError: If download fails or returns non-200 status.
    """
    dest = Path(destination_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(f"{dest.suffix}.tmp")

    logger.info(f"Downloading model artifact from {url} to {dest}...")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(temp_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        # Atomic rename once download is completely finished
        temp_dest.replace(dest)
        logger.info(f"Successfully saved checkpoint to: {dest} ({dest.stat().st_size:,} bytes)")
        return dest
    except Exception as exc:
        if temp_dest.exists():
            temp_dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download model weights from '{url}': {exc}"
        ) from exc


def get_configured_model_url() -> Optional[str]:
    """Retrieve model URL from environment variables or Streamlit secrets."""
    # 1. Environment variables
    for env_var in ["MEDVISION_MODEL_URL", "MODEL_URL"]:
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()

    # 2. Streamlit secrets (if available)
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            for secret_key in ["MEDVISION_MODEL_URL", "MODEL_URL"]:
                if secret_key in st.secrets and str(st.secrets[secret_key]).strip():
                    return str(st.secrets[secret_key]).strip()
    except Exception:
        # st.secrets might raise if not configured
        pass

    return None


def resolve_model_checkpoint(
    checkpoint_path: Optional[Union[str, Path]] = None,
    target_dir: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve or acquire the Stage 2 DenseNet121 model checkpoint.

    Resolution Strategy:
    1. Explicit checkpoint_path argument if provided and exists.
    2. Environment variable `MEDVISION_MODEL_PATH` or `MODEL_PATH` if exists.
    3. Standard local search paths:
       - <root>/final_artifacts/densenet121_stage2_best.keras
       - <root>/models/checkpoints/densenet121_stage2_best.keras
       - <root>/artifacts/models/densenet121_stage2_best.keras
       - ~/.cache/medvision/densenet121_stage2_best.keras
    4. Remote acquisition via `MEDVISION_MODEL_URL` or `MODEL_URL`:
       - Download into local cache directory on first startup.
    5. If unresolvable, raise FileNotFoundError with actionable instructions.

    Args:
        checkpoint_path: Optional explicit path override.
        target_dir: Optional target directory for downloaded weights.
        root_dir: Optional project root directory override.

    Returns:
        Path to existing verified model checkpoint.

    Raises:
        FileNotFoundError: If checkpoint cannot be located or downloaded.
    """
    root = Path(root_dir).resolve() if root_dir else Path(__file__).resolve().parent.parent.parent

    # 1. Explicit path
    if checkpoint_path is not None:
        p = Path(checkpoint_path).resolve()
        if p.exists():
            return p
        logger.warning(f"Explicitly provided checkpoint_path not found: {p}")

    # 2. Environment path override
    for env_var in ["MEDVISION_MODEL_PATH", "MODEL_PATH"]:
        env_val = os.environ.get(env_var)
        if env_val and env_val.strip():
            env_p = Path(env_val.strip()).resolve()
            if env_p.exists():
                return env_p

    # 3. Standard local search locations
    local_candidates = [
        root / "final_artifacts" / DEFAULT_CHECKPOINT_FILENAME,
        root / "models" / "checkpoints" / DEFAULT_CHECKPOINT_FILENAME,
        root / "artifacts" / "models" / DEFAULT_CHECKPOINT_FILENAME,
        Path.home() / ".cache" / "medvision" / DEFAULT_CHECKPOINT_FILENAME,
    ]

    for candidate in local_candidates:
        if candidate.exists():
            logger.info(f"Found local model checkpoint at: {candidate}")
            return candidate

    # 4. Remote URL download strategy
    model_url = get_configured_model_url()
    if model_url:
        download_dest = (
            Path(target_dir).resolve() / DEFAULT_CHECKPOINT_FILENAME
            if target_dir
            else root / "models" / "checkpoints" / DEFAULT_CHECKPOINT_FILENAME
        )
        if download_dest.exists():
            return download_dest
        return download_model_checkpoint(model_url, download_dest)

    # 5. Fail with actionable error
    searched_paths = "\n".join(f" - {p.resolve()}" for p in local_candidates)
    raise FileNotFoundError(
        f"DenseNet121 Stage 2 checkpoint ('{DEFAULT_CHECKPOINT_FILENAME}') not found.\n"
        f"Searched local paths:\n{searched_paths}\n\n"
        "Deployment Resolution Options:\n"
        "1. For Streamlit Community Cloud: Set 'MODEL_URL' or 'MEDVISION_MODEL_URL' in App Settings -> Secrets:\n"
        '   MODEL_URL = "https://github.com/SwastikPandey1024/MedVision-AI/releases/download/v0.1.0-alpha/densenet121_stage2_best.keras"\n'
        "2. For Local Runtime: Place 'densenet121_stage2_best.keras' into 'final_artifacts/' or 'models/checkpoints/'.\n"
        "3. Set 'MODEL_PATH' environment variable pointing to the checkpoint file."
    )


@st.cache_resource(show_spinner="Acquiring and loading DenseNet121 model weights...")
def get_cached_model(checkpoint_path: Optional[str] = None) -> Tuple[keras.Model, str]:
    """Load and cache DenseNet121 model and detect target conv layer."""
    ckpt_path = resolve_model_checkpoint(checkpoint_path)
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
