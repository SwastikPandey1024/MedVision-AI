# 🌐 MedVision-AI: Phase 9 — FastAPI REST API Service

> **Documentation, schema specifications, and integration guide for the MedVision-AI REST API service providing pneumonia detection and Grad-CAM visual explainability.**

---

## 📌 1. Service Overview

The MedVision-AI REST API exposes the validated **DenseNet121** model ($t=0.60$ frozen decision threshold) and the **Grad-CAM** saliency engine through high-performance asynchronous HTTP endpoints built with **FastAPI** and **Pydantic**.

### Core Engineering Principles
- **Strict Non-Clinical Terminology:** Outputs are explicitly identified as statistical model predictions / probabilities, not diagnostic advice.
- **In-Memory Zero-Storage Lifecycle:** Image payloads (DICOM, PNG, JPEG) are processed in volatile memory and immediately discarded.
- **Payload Guardrails:** Enforces file-size limits (25 MB max), magic-byte verification, and graceful error responses.
- **Asynchronous Architecture:** High-throughput async request handling with CORS support.

---

## 🛣️ 2. API Endpoints Specification

### 1. `GET /health`
Returns service health, model loading status, target layer, and device.

**Sample Response (`200 OK`):**
```json
{
  "status": "healthy",
  "service": "MedVision-AI REST API",
  "model_loaded": true,
  "model_name": "DenseNet121",
  "frozen_threshold": 0.60,
  "version": "1.0.0",
  "device": "CPU"
}
```

---

### 2. `GET /metadata`
Returns model architecture parameters, input resolution, target layer, and verified held-out test metrics.

**Sample Response (`200 OK`):**
```json
{
  "model_architecture": "DenseNet121",
  "total_parameters": 7301185,
  "frozen_threshold": 0.60,
  "input_resolution": "224x224x3",
  "target_conv_layer": "conv5_block16_2_conv",
  "validation_metrics": {
    "dataset": "RSNA Validation Split (4,003 unique patients)",
    "roc_auc": 0.8358,
    "pr_auc": 0.5944,
    "f1_score_at_0.60": 0.5898,
    "specificity_at_0.60": 0.8384
  },
  "held_out_test_metrics": {
    "dataset": "RSNA Held-Out Test Split (4,003 unique patients)",
    "roc_auc": 0.8381,
    "pr_auc": 0.6022,
    "f1_score_at_0.60": 0.5908,
    "accuracy_at_0.60": 0.7979,
    "sensitivity_at_0.60": 0.6475,
    "specificity_at_0.60": 0.8417,
    "precision_at_0.60": 0.5433,
    "patient_leakage": "0.0%"
  },
  "non_clinical_disclaimer": "MedVision-AI is an academic research and educational demonstration tool..."
}
```

---

### 3. `POST /predict`
Uploads a radiograph (`multipart/form-data`) and returns model probability and thresholded classification.

**Parameters:**
- `file` (Form file): Image file (`.dcm`, `.png`, `.jpg`, `.jpeg`).
- `threshold` (Query float, optional, default: `0.60`): Classification operating threshold.

**Sample Response (`200 OK`):**
```json
{
  "pneumonia_probability": 0.2719,
  "predicted_class": "Normal",
  "is_pneumonia": false,
  "decision_threshold": 0.60,
  "image_format": "DICOM",
  "inference_time_ms": 42.5,
  "model_version": "DenseNet121-Stage2-v1.0",
  "non_clinical_disclaimer": "MedVision-AI is an academic research and educational demonstration tool..."
}
```

---

### 4. `POST /explain`
Uploads a radiograph and returns Grad-CAM normalized heatmap and blended radiograph overlay as Base64 PNG data strings.

**Parameters:**
- `file` (Form file): Image file.
- `alpha` (Query float, optional, default: `0.40`): Heatmap blend factor.
- `target_layer` (Query string, optional): Target convolutional layer override.

**Sample Response (`200 OK`):**
```json
{
  "pneumonia_probability": 0.7412,
  "predicted_class": "Pneumonia",
  "decision_threshold": 0.60,
  "target_layer": "conv5_block16_2_conv",
  "heatmap_base64": "data:image/png;base64,iVBORw0KGgoAAA...",
  "overlay_base64": "data:image/png;base64,iVBORw0KGgoAAA...",
  "inference_time_ms": 85.3,
  "non_clinical_disclaimer": "..."
}
```

---

### 5. `POST /predict-and-explain`
Unified endpoint performing both classification inference and side-by-side visual explanation.

---

## 🚀 3. Running the Server

```bash
# Start API server using uvicorn
uvicorn medvision.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI Swagger docs will be available at `http://localhost:8000/docs`.

---

## 🧪 4. Automated Testing

Verified by `tests/test_api.py`:
```bash
python -m pytest tests/test_api.py -q
```
