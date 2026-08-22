"""Integration tests for MedVision-AI FastAPI REST API service."""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import keras

from medvision.api.main import app
from medvision.api.services import ModelService


@pytest.fixture(scope="module")
def client():
    """Create TestClient with initialized ModelService."""
    service = ModelService.get_instance()
    # If production checkpoint not already loaded, create dummy model for testing
    if not service.is_loaded():
        inputs = keras.layers.Input(shape=(224, 224, 3), name="test_in")
        x = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv5_block16_2_conv")(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)
        service.model = keras.Model(inputs=inputs, outputs=outputs, name="MockDenseNet")
        service.target_layer = "conv5_block16_2_conv"
        service.device = "CPU (Mock)"

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_png_bytes():
    """Generate in-memory valid PNG radiograph bytes."""
    img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_get_health_endpoint(client):
    """Test /health returns valid status, model loaded flag, and threshold 0.60."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "MedVision-AI REST API"
    assert data["model_loaded"] is True
    assert data["frozen_threshold"] == 0.60
    assert "version" in data


def test_get_metadata_endpoint(client):
    """Test /metadata returns architectural and validated test metrics."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_architecture"] == "DenseNet121"
    assert data["total_parameters"] == 7301185
    assert data["frozen_threshold"] == 0.60
    assert "held_out_test_metrics" in data
    assert data["held_out_test_metrics"]["roc_auc"] == 0.8381
    assert data["held_out_test_metrics"]["pr_auc"] == 0.6022
    assert "non_clinical_disclaimer" in data


def test_predict_endpoint_valid_png(client, sample_png_bytes):
    """Test /predict returns probability and predicted class."""
    files = {"file": ("radiograph.png", sample_png_bytes, "image/png")}
    response = client.post("/predict?threshold=0.60", files=files)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["pneumonia_probability"] <= 1.0
    assert data["predicted_class"] in ["Pneumonia", "Normal"]
    assert isinstance(data["is_pneumonia"], bool)
    assert data["decision_threshold"] == 0.60
    assert "inference_time_ms" in data
    assert "non_clinical_disclaimer" in data


def test_explain_endpoint(client, sample_png_bytes):
    """Test /explain returns Grad-CAM base64 encoded heatmap and overlay."""
    files = {"file": ("chest.png", sample_png_bytes, "image/png")}
    response = client.post("/explain?alpha=0.4", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "heatmap_base64" in data
    assert "overlay_base64" in data
    assert data["heatmap_base64"].startswith("data:image/png;base64,")
    assert data["overlay_base64"].startswith("data:image/png;base64,")
    assert data["target_layer"] == "conv5_block16_2_conv"


def test_predict_and_explain_endpoint(client, sample_png_bytes):
    """Test /predict-and-explain unified endpoint."""
    files = {"file": ("chest_study.png", sample_png_bytes, "image/png")}
    response = client.post("/predict-and-explain?threshold=0.60&alpha=0.4", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "pneumonia_probability" in data
    assert "overlay_base64" in data
    assert "comparison_base64" in data
    assert data["comparison_base64"].startswith("data:image/png;base64,")


def test_predict_empty_file_returns_400(client):
    """Test uploading 0-byte file returns 400 Bad Request."""
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_corrupt_bytes_returns_400(client):
    """Test uploading corrupted bytes returns 400 Bad Request."""
    files = {"file": ("corrupt.png", b"not-a-valid-image-stream", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "failed to decode" in response.json()["detail"].lower()
