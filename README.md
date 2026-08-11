# MedVision-AI: Explainable Chest X-Ray Pneumonia Detection System

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Status](https://img.shields.io/badge/CI-passing-brightgreen.svg)](#)

An enterprise-grade, end-to-end, explainable deep-learning research application for detecting pneumonia from chest X-ray images.

> [!IMPORTANT]
> **Non-Clinical Software Disclaimer**: MedVision-AI is an educational and AI research demonstration tool. It is **not** a certified medical device and must **not** be used for clinical diagnostic decision-making or patient management.

---

## 💻 Hardware-Aware MLOps Architecture

MedVision-AI is architected for a **Hybrid Local CPU / Cloud GPU Workflow**:

- **Local Laptop (Development & Control Environment)**:
  - **Hardware:** Intel Core i5-13500H CPU, 16 GB RAM, integrated graphics (CPU-only execution).
  - **Role:** Pipeline authoring, REST API & Streamlit development, unit testing, and fast CPU smoke tests (`execution_mode: "development"`).
- **Cloud GPU (Training Environment - Kaggle / Google Colab)**:
  - **Hardware:** Cloud NVIDIA Tesla T4 / P100 GPU.
  - **Role:** Full dataset training, transfer learning, fine-tuning, and hyperparameter sweeps (`execution_mode: "full"`).
- **GitHub Repository**: Single source of truth for all source code, tests, and configuration. Model weights are versioned in external artifact storage and never committed to git.

---

## 📌 Key Architectural Highlights

- **Extensible Architecture Selection**: Model factory supporting **DenseNet121** (primary candidate), **EfficientNetB0**, and **Custom CNN Baseline**.
- **Patient-Aware Group Splitting**: Mandatory `PatientID`-based partitioning to prevent data leakage between train/val/test splits.
- **Dynamic Visual Explainability**: Grad-CAM heatmaps with automatic convolutional layer target detection.
- **Decoupled Architecture**: Independent Flask REST API (`/health`, `/predict`) served separately from an interactive Streamlit UI dashboard.

---

## 🗺️ 13-Phase Implementation Roadmap

- [x] **Phase 0: Foundational Hardware-Aware Architecture Setup** (Current)
- [ ] **Phase 1: Dataset Acquisition & Quality Control EDA**
- [ ] **Phase 2: Preprocessing Engine & `tf.data` Pipeline**
- [ ] **Phase 3: Custom CNN Baseline Model**
- [ ] **Phase 4: DenseNet121 Transfer Learning Architecture**
- [ ] **Phase 5: Controlled Layer Fine-Tuning**
- [ ] **Phase 6: Systematic Hyperparameter & Data Experiments**
- [ ] **Phase 7: Comprehensive Evaluation & Error Analysis**
- [ ] **Phase 8: Grad-CAM Explainability Engine**
- [ ] **Phase 9: Flask REST API Backend**
- [ ] **Phase 10: Streamlit Interactive Web Application**
- [ ] **Phase 11: Docker Containerization & Cloud Deployment**
- [ ] **Phase 12: End-to-End Testing & Comprehensive Documentation**
- [ ] **Phase 13: GitHub Showcase & Project Release**

---

## 📁 Standardized Directory Structure

```
MedVision-AI/
├── .github/workflows/      # CI/CD automation pipelines
├── data/
│   ├── raw/                # Unzipped DICOM/PNG images (git-ignored)
│   ├── processed/          # Normalized tensors (git-ignored)
│   └── metadata/           # Patient manifest CSVs (git-ignored)
├── docs/                   # Engineering & Product Docs (PRD, BRD, SRS, ADRs)
├── models/
│   ├── checkpoints/        # Intermediate training checkpoints (git-ignored)
│   └── production/         # Exported production model artifacts (git-ignored)
├── artifacts/
│   ├── tensorboard/        # TensorBoard training logs (git-ignored)
│   └── experiments/        # Metrics & confusion matrices (git-ignored)
├── src/medvision/          # Core Python Package
│   ├── config/             # YAML & settings loaders
│   ├── data/               # Loaders, preprocessors, patient splitters
│   ├── models/             # Architecture factory & trainers
│   ├── explainability/     # Grad-CAM heatmap engine
│   ├── evaluation/         # Clinical & ML metrics calculation
│   ├── api/                # Flask REST API server
│   ├── ui/                 # Streamlit web dashboard
│   └── utils/              # Logging, seed setter, hardware auto-detector
├── tests/                  # Pytest unit & integration tests
├── pyproject.toml          # PEP 517 build & tool configuration
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development & test dependencies
├── LICENSE                 # MIT License & Medical Disclaimer
└── README.md               # Project homepage
```

---

## 🚀 Quickstart Guide

### 1. Local Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/MedVision-AI.git
cd MedVision-AI

# Create virtual environment (Python 3.11)
python -m venv .venv
# Activate: Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate

# Install editable package with dev dependencies
pip install -e ".[dev]"
```

### 2. Run Test Suite
```bash
python -m pytest tests/
```
