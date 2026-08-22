# 🩺 MedVision-AI: Explainable Chest X-Ray Pneumonia Detection System

[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Deep Learning Framework](https://img.shields.io/badge/TensorFlow-2.16%2B%20%7C%20Keras%203-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Dataset: RSNA](https://img.shields.io/badge/Dataset-RSNA%20Pneumonia%20(26.6k)-blue.svg)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
[![Tests Passing](https://img.shields.io/badge/Tests-92%2F92%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](#-automated-testing--reproducibility)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](api/main.py)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](app/streamlit_app.py)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, leakage-free deep learning system for pneumonia detection from frontal chest radiographs. Built with **DenseNet121** transfer learning, multi-GPU mixed-precision training, numerical stability forensics, validation-isolated decision threshold optimization, **Grad-CAM** visual interpretability, **FastAPI** asynchronous serving, and an interactive **Streamlit** diagnostic dashboard.

> [!CAUTION]
> **Non-Clinical Research & Education Disclaimer**: MedVision-AI is an academic machine learning research and educational demonstration tool. It is **not** a certified medical device (FDA/EMA) and must **never** be used for clinical diagnosis, patient screening, or medical decision-making. See [`docs/disclaimer.md`](docs/disclaimer.md).

---

## 🏆 Final Verified Test Performance (4,003 Held-Out Patients)

Evaluated on the **official held-out test split of 4,003 unique patients (0.0% patient leakage)** from the RSNA Pneumonia Detection Challenge at the **frozen optimal decision threshold ($t = 0.60$)** determined exclusively on validation data:

| Metric | Score | Clinical / Engineering Significance |
| :--- | :---: | :--- |
| **ROC-AUC (Primary)** | **`0.8381`** | High global discriminability across all false-positive operating rates |
| **PR-AUC (Primary)** | **`0.6022`** | Robust positive-class precision/recall trade-off under 22.5% prevalence |
| **Specificity** | **`84.17%`** | **161 fewer false alarms** compared to default 0.50 threshold (2,610 / 3,101 True Negatives) |
| **Accuracy** | **`79.79%`** | Overall classification accuracy across 4,003 unseen patients |
| **Sensitivity / Recall** | **`64.75%`** | True positive detection rate (584 / 902 confirmed pneumonia cases detected) |
| **Precision** | **`54.33%`** | Positive predictive value (584 TP / 1,075 total flagged cases) |
| **F1-Score** | **`0.5908`** | Harmonic balance between precision and recall at operating point |

---

## 📑 Project Navigation & Documentation

- 📖 **[Portfolio Project Page](docs/portfolio.md)** — Comprehensive case study and technical narrative.
- 🏗️ **[System Architecture](docs/architecture.md)** — Detailed Mermaid diagram and subsystem breakdown.
- 📊 **[Final Metrics Table](docs/final_metrics.md)** — Complete split-by-split validation and test performance breakdown.
- 🔬 **[Grad-CAM Saliency Engine](docs/phase8_gradcam.md)** — Mathematical foundation and explainability guide.
- 🌐 **[FastAPI REST Service](docs/phase9_api.md)** — OpenAPI endpoint specifications and schema documentation.
- 🩺 **[Streamlit Radiograph UI](docs/phase10_streamlit.md)** — Interactive dashboard operation and layout guide.
- 🐳 **[Docker & Cloud Deployment](docs/phase11_deployment.md)** — Container build and cloud-readiness recipes.
- 🧪 **[Integration & Load Test Report](docs/phase12_integration_loadtest.md)** — Measured concurrency and latency benchmarks.
- 🎙️ **[Interview Demo Scripts](docs/portfolio/demo_script.md)** — 30-second, 2-minute, and 5-minute technical narratives.
- 🔄 **[Reproducibility Guide](docs/reproducibility.md)** — Environment configuration, GPU training, and CLI execution.
- 📄 **[Resume Project Bullets](docs/resume_bullets.md)** — ATS-friendly and technical resume bullets.
- 💼 **[LinkedIn Descriptions](docs/linkedin_description.md)** — Recruiter-friendly posts and project descriptions.
- 🖼️ **[Screenshot Catalog](docs/screenshot_gallery.md)** — Portfolio screenshot gallery and capture specifications.
- ⚠️ **[Non-Clinical Disclaimer](docs/disclaimer.md)** — Full regulatory and medical scope disclaimer.

---

## 🎯 Problem Statement & Medical Context

Pneumonia accounts for over 2.5 million deaths annually worldwide. While chest radiography (CXR) is the standard initial imaging modality for detection, subtle lung opacities and high clinical workloads create significant triage bottlenecks.

Developing reliable machine learning models for chest X-rays requires overcoming critical engineering challenges:
1. **Patient Memorization (Data Leakage):** Hospital datasets frequently contain multiple radiographs per patient. Splitting datasets at the image level causes models to memorize patient ribcage geometry rather than pathology, creating misleadingly high benchmark numbers.
2. **Class Imbalance:** Pneumonia is present in 22.5% of screening images in the RSNA cohort, making raw accuracy deceptive and requiring PR-AUC and ROC-AUC as primary metrics.
3. **Fine-Tuning Instability:** Fine-tuning deep backbones with small batches often corrupts pre-trained Batch Normalization running statistics, destabilizing loss convergence.
4. **Threshold Snooping:** Tuning classification thresholds directly on test data leads to severe overfitting and inflated claims.

---

## 🏗️ System Architecture & Engineering Flow

```
RSNA Dataset (26,684 CXRs)
    │
    ▼
Manifest Parsing & Bounding-Box QA
    │
    ▼
Target-Stratified Group Splitting (patient_id) ──► 0.0% Patient Leakage
    │
    ├─── Train Split (70%): 18,678 Patients
    ├─── Validation Split (15%): 4,003 Patients
    └─── Held-Out Test Split (15%): 4,003 Patients
    │
    ▼
tf.data Parallel Processing Engine (Spatial Augmentation, Normalization, Prefetching)
    │
    ▼
DenseNet121 Staged Training Lifecycle:
    ├─── Stage 1: Feature Extraction (Backbone Frozen, Adam LR=1e-4, Clipnorm=1.0)
    └─── Stage 2: Controlled Fine-Tuning (Top 20 Layers Unfrozen, BatchNorm FROZEN, LR=1e-5)
    │
    ▼
Numerical Forensic Analysis (FP32 vs. mixed_float16 Stability Profiling)
    │
    ▼
Zero-Leakage Evaluation Pipeline:
    ├─── Validation Split Inference & 81-Point Threshold Search (t=0.60 selected)
    └─── Frozen Threshold (t=0.60) Applied to 4,003-Patient Test Split
    │
    ▼
Production Deployment & Interpretability:
    ├─── Grad-CAM Visual Saliency Engine (conv5_block16_2_conv)
    ├─── FastAPI Asynchronous REST Service (Port 8000)
    ├─── Streamlit Interactive Dashboard (Port 8501)
    └─── Production Docker Containerization (Python 3.11-slim, non-root)
```

### 1. Patient-Aware Group Splitting (0% Leakage)
- **Train Set (70%):** `18,678` unique patients
- **Validation Set (15%):** `4,003` unique patients
- **Held-Out Test Set (15%):** `4,003` unique patients
- **Leakage Verification:** Automated audit confirms **`0.0%` patient intersection** across all partitions.

### 2. Two-Stage Transfer Learning & BatchNorm Protection
- **Backbone:** DenseNet121 (7,301,185 parameters) pre-trained on ImageNet.
- **Classification Head:** Global Average Pooling $\to$ Batch Normalization $\to$ Dropout ($p=0.4$) $\to$ Dense (128 units, ReLU) $\to$ Dropout ($p=0.2$) $\to$ Dense (1 unit, Sigmoid).
- **Stage 1 (Feature Extraction):** Backbone frozen; classification head trained for 5 epochs with Adam ($LR = 10^{-4}$) on 2 × Tesla T4 GPUs with `mixed_float16`.
- **Stage 2 (Fine-Tuning):** Top 20 convolutional layers unfrozen ($LR = 10^{-5}$). **Critical BatchNorm Policy:** All Batch Normalization layers were strictly locked (`trainable = False`) to prevent small-batch noise from corrupting pre-trained channel statistics.

### 3. Zero-Leakage Decision Threshold Optimization
- 81 candidate thresholds ($0.10$ to $0.90$, $\Delta t = 0.01$) evaluated **exclusively on validation predictions**.
- Optimal operating point selected: **$t = 0.60$** (maximizing validation F1-score).
- Audit trail generated: `test_data_used: False` verified before evaluating test data.

---

## 🔬 Visual Explainability (Grad-CAM)

MedVision-AI integrates a post-hoc Grad-CAM engine computing target class gradients with respect to `conv5_block16_2_conv`:

```bash
# Generate Grad-CAM visualization from CLI
python scripts/gradcam.py \
  --checkpoint final_artifacts/densenet121_stage2_best.keras \
  --image test_dl.dcm \
  --threshold 0.60 \
  --output-dir artifacts/explainability
```

Generates normalized heatmaps, alpha-blended overlays ($\alpha=0.40$), and side-by-side comparative panels.

---

## 🌐 FastAPI REST API Service

Asynchronous, typed REST service with Pydantic validation:

```bash
# Start API server
uvicorn medvision.api.main:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service status, active model, device, and frozen threshold |
| `GET` | `/metadata` | Architectural specs, parameters (7.3M), and held-out test benchmarks |
| `POST` | `/predict` | Ingests radiograph, returns pneumonia probability & binary flag |
| `POST` | `/explain` | Returns Base64-encoded Grad-CAM heatmap and blended overlay |
| `POST` | `/predict-and-explain` | Unified prediction and visual explanation response |

---

## 🩺 Streamlit Interactive Radiologist Dashboard

```bash
# Launch Streamlit web application
streamlit run app/streamlit_app.py --server.port 8501
```

Features:
- **Direct DICOM Parsing:** Native 16-bit DICOM ingestion with VOI LUT windowing and fallback percentile clipping.
- **Interactive Threshold Slider:** Dynamically inspect sensitivity vs. specificity operating trade-offs.
- **Side-by-Side Saliency Visualizer:** Inspect original radiograph alongside Grad-CAM focal opacity localization.
- **Persistent Non-Clinical Disclaimer:** Transparent educational framing.

---

## 🐳 Docker Deployment

```bash
# Build production container
docker build -t medvision-ai:latest .

# Run REST API (Port 8000)
docker run -d -p 8000:8000 --name medvision-api medvision-ai:latest

# Run with Docker Compose (API + UI)
docker compose up -d
```

---

## 📊 Comprehensive Evaluation Results

| Dataset Partition | Total Patients | Decision Threshold | PR-AUC (Primary) | ROC-AUC | F1-Score | Specificity | Sensitivity | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Split** | 4,003 | 0.50 (Default) | `0.5944` | `0.8358` | `0.5836` | `78.30%` | `71.95%` | `49.09%` | `76.87%` |
| **Validation Split** | 4,003 | 0.60 (Optimal) | `0.5944` | `0.8358` | `0.5898` | `83.84%` | `64.08%` | `54.63%` | `79.39%` |
| **Held-Out Test Split** | 4,003 | 0.50 (Reference) | `0.6023` | `0.8380` | `0.5962` | `78.97%` | `73.17%` | `50.30%` | `77.67%` |
| **Held-Out Test Split** ⭐ | **4,003** | **0.60 (FROZEN)** | **`0.6022`** | **`0.8381`** | **`0.5908`** | **`84.17%`** | **`64.75%`** | **`54.33%`** | **`79.79%`** |

### Confusion Matrix on Held-Out Test Set ($N = 4,003$ Patients @ Frozen $t = 0.60$):
```
                        Actual Normal (3,101)    Actual Pneumonia (902)
Predicted Normal (2,928):      2,610 (TN)                318 (FN)
Predicted Pneumonia (1,075):     491 (FP)                584 (TP)
```
- **True Positives (TP):** `584` | **True Negatives (TN):** `2,610`
- **False Positives (FP):** `491` (Reduced by 161 compared to $t=0.50$) | **False Negatives (FN):** `318`

---

## ⚡ Automated Testing & Reproducibility

```bash
# 1. Setup virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run automated test suite (92 tests)
python -m pytest tests/ -q

# 3. Execute concurrent load testing
python scripts/load_test.py --endpoint /health --requests 50 --concurrency 5
```

---

## 📁 Repository Structure

```
MedVision-AI/
├── api/                        # REST API package entrypoint
├── app/                        # Streamlit web application
│   ├── components/             # Header, disclaimer, metrics card components
│   ├── services/               # Caching and inference service
│   └── streamlit_app.py        # Streamlit dashboard layout
├── artifacts/                  # Local evaluation metrics, audit JSONs, plots, Grad-CAMs
│   ├── evaluation/             # Confusion matrices, ROC/PR curves, threshold audit reports
│   └── explainability/         # Generated Grad-CAM visual assets
├── docs/                       # Project documentation & portfolio assets
│   ├── architecture.md         # System architecture & Mermaid diagrams
│   ├── final_metrics.md        # Comprehensive verified metrics breakdown
│   ├── phase8_gradcam.md       # Grad-CAM engine design & math
│   ├── phase9_api.md           # FastAPI route specifications
│   ├── phase10_streamlit.md    # Streamlit UI architecture
│   ├── phase11_deployment.md   # Docker & multi-cloud deployment guide
│   ├── phase12_integration_loadtest.md # Load test benchmarks
│   ├── phase_status.md         # 13-phase implementation registry
│   ├── portfolio/              # Recruiter, architecture, and demo scripts
│   ├── reproducibility.md      # Step-by-step reproduction guide
│   └── screenshot_gallery.md   # Visual asset inventory
├── final_artifacts/            # Validated model checkpoints (local only)
├── scripts/                    # Command-line entry points
│   ├── evaluate.py             # Multi-split evaluation & threshold engine
│   ├── gradcam.py              # Grad-CAM saliency CLI runner
│   ├── load_test.py            # Concurrent benchmark runner
│   └── train.py                # Two-stage training lifecycle runner
├── src/medvision/              # Core application package
│   ├── api/                    # FastAPI routers, schemas, services
│   ├── data/                   # DICOM/PNG ingestion, group splitting, tf.data pipeline
│   ├── evaluation/             # Metrics, ROC/PR plotting, threshold search
│   ├── explainability/         # Grad-CAM saliency mapping engine
│   ├── models/                 # Custom CNN baseline & DenseNet121 architecture
│   └── utils/                  # Model loader, custom metrics, logging, forensics
├── tests/                      # Automated unit & integration test suite (92 tests)
├── Dockerfile                  # Production container definition (Python 3.11-slim)
└── docker-compose.yml          # Multi-service local orchestration
```

---

## 🗺️ 13-Phase Implementation Roadmap

- [x] **Phase 0: Foundational Hardware-Aware Architecture Setup**
- [x] **Phase 1: Dataset Acquisition & Quality Control EDA**
- [x] **Phase 2: Preprocessing Engine & `tf.data` Pipeline**
- [x] **Phase 3: Custom CNN Baseline & Training Engine**
- [x] **Phase 4: DenseNet121 Transfer Learning Fine-Tuning & Multi-GPU Strategy**
- [x] **Phase 5: Controlled Layer Fine-Tuning & BatchNorm Protection**
- [x] **Phase 6: Systematic Hyperparameter & Numerical Forensics**
- [x] **Phase 7: Comprehensive Zero-Leakage Evaluation & Threshold Selection**
- [x] **Phase 8: Grad-CAM Saliency & Visual Explainability Engine**
- [x] **Phase 9: FastAPI REST API Service**
- [x] **Phase 10: Streamlit Interactive Radiograph Diagnostic UI**
- [x] **Phase 11: Docker Containerization & Cloud-Ready Deployment**
- [x] **Phase 12: End-to-End Integration & Load Testing**
- [x] **Phase 13: Open-Source Community Release & Paper Documentation**

---

## ⚠️ Limitations

1. **Frontal Adult Radiographs Only:** The model was trained exclusively on adult posteroanterior (PA) and anteroposterior (AP) chest radiographs from the RSNA dataset. It has not been validated on pediatric, lateral, or bedside emergency views.
2. **Binary Classification Scope:** The model detects the presence of pneumonia-associated lung opacities vs. normal/non-opacity findings, but does not differentiate viral, bacterial, or fungal etiologies.
3. **Hardware Context:** Multi-GPU training was benchmarked on 2 × NVIDIA Tesla T4 GPUs with mixed precision; local CPU evaluation uses the deterministic development subset loader.

---

## 📜 License & Citation

This project is open-source under the [MIT License](LICENSE).

```bibtex
@misc{medvision_ai_2026,
  author = {Swastik Pandey},
  title = {MedVision-AI: Explainable Chest X-Ray Pneumonia Detection System},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SwastikPandey1024/MedVision-AI}}
}
```
