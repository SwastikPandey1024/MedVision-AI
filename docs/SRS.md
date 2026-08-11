# Software Requirements Specification (SRS)

## Project: MedVision-AI
**Document Version:** 1.0  
**Specification Standard:** IEEE 830 / ISO/IEC/IEEE 29148  

---

## 1. Functional Requirements

### FR-01: Data Processing & Splitting
- **FR-01.1:** The system shall parse RSNA Chest X-Ray metadata index files (`stage_2_train_labels.csv`).
- **FR-01.2:** The system shall split records into 70% Train, 15% Validation, and 15% Test sets grouped strictly by `patient_id`.
- **FR-01.3:** The system shall stream images via `tf.data.Dataset` pipelines with prefetching and batching.

### FR-02: Model Architectures & Training
- **FR-02.1:** The system shall provide a baseline Custom CNN architecture for comparison.
- **FR-02.2:** The system shall instantiate DenseNet121 pre-trained on ImageNet with customizable dense classification heads.
- **FR-02.3:** The system shall support multi-stage training (Phase 4 feature extraction followed by Phase 5 layer unfreezing).

### FR-03: Explainability Engine
- **FR-03.1:** The system shall calculate Grad-CAM activation heatmaps from target convolutional layers (`conv5_block16_concat`).
- **FR-03.2:** The system shall superimpose the generated heatmap on the input X-ray image with configurable alpha opacity.

### FR-04: REST API Backend
- **FR-04.1:** The API shall provide a `GET /health` endpoint returning system health and model loading status.
- **FR-04.2:** The API shall provide a `POST /predict` endpoint accepting multipart form image uploads (`.png`, `.jpg`, `.jpeg`, `.dcm`).
- **FR-04.3:** The API shall validate file payloads and return structured JSON containing predicted class, probability, confidence level, and base64-encoded Grad-CAM image.

### FR-05: Streamlit Frontend Dashboard
- **FR-05.1:** The UI shall allow drag-and-drop X-ray file uploads.
- **FR-05.2:** The UI shall display side-by-side original image vs Grad-CAM heatmap visualization.
- **FR-05.3:** The UI shall display model confidence metrics and non-clinical medical disclaimers.

---

## 2. Non-Functional Requirements

### NFR-01: Performance & Efficiency
- Model inference on single image payload shall complete in $< 1.5$ seconds on standard CPU.
- `tf.data` input pipeline shall utilize parallel prefetching to eliminate GPU starvation.

### NFR-02: Security & File Validation
- File upload payloads shall be restricted to a maximum of 16 MB.
- Uploaded content shall be validated for file extension and magic byte header before decoding.

### NFR-03: Portability & Containerization
- The Flask REST API service shall be fully containerizable via Docker and compatible with cloud app environments.
