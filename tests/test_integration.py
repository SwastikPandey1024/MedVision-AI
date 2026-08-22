"""End-to-end integration and resilience test suite for MedVision-AI."""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from medvision.api.main import app
from medvision.api.services import ModelService, decode_image_bytes, MAX_FILE_SIZE_BYTES
from medvision.explainability.gradcam import generate_gradcam_explanation


@pytest.fixture(scope="module")
def client():
    service = ModelService.get_instance()
    service.initialize()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_radiograph_bytes():
    img = np.random.randint(50, 200, size=(224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_e2e_inference_to_explainability_pipeline(client, valid_radiograph_bytes):
    """Verify complete end-to-end workflow: decode -> predict -> Grad-CAM -> response payload."""
    files = {"file": ("patient_cxr.png", valid_radiograph_bytes, "image/png")}
    response = client.post("/predict-and-explain?threshold=0.60&alpha=0.40", files=files)

    assert response.status_code == 200
    data = response.json()

    # Verify prediction metadata
    assert "pneumonia_probability" in data
    assert 0.0 <= data["pneumonia_probability"] <= 1.0
    assert data["predicted_class"] in ["Pneumonia", "Normal"]
    assert data["decision_threshold"] == 0.60

    # Verify Grad-CAM visual components
    assert "overlay_base64" in data
    assert "comparison_base64" in data
    assert data["overlay_base64"].startswith("data:image/png;base64,")
    assert data["comparison_base64"].startswith("data:image/png;base64,")

    # Verify disclaimer is attached
    assert "non_clinical_disclaimer" in data


def test_edge_case_empty_file_upload(client):
    """Test empty (0-byte) payload returns HTTP 400."""
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_edge_case_oversized_file_upload():
    """Test files exceeding MAX_FILE_SIZE_BYTES trigger ValueError in decoding."""
    huge_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
    with pytest.raises(ValueError) as excinfo:
        decode_image_bytes(huge_bytes, "huge.png")
    assert "exceeds maximum allowed limit" in str(excinfo.value)


def test_edge_case_unsupported_corrupted_format(client):
    """Test corrupted non-image binary stream returns HTTP 400."""
    garbage = b"\x00\x01\x02\x03\x04INVALID_DATA"
    files = {"file": ("sample.raw", garbage, "application/octet-stream")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_missing_file_parameter_returns_422(client):
    """Test omitting the file field returns HTTP 422 Unprocessable Entity."""
    response = client.post("/predict")
    assert response.status_code == 422
