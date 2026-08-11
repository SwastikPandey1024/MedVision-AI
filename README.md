# MedVision-AI: Explainable Chest X-Ray Pneumonia Detection System

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org)
[![Dataset: RSNA](https://img.shields.io/badge/Dataset-RSNA%20Pneumonia-blue.svg)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Status](https://img.shields.io/badge/CI-passing-brightgreen.svg)](#)

An enterprise-grade, end-to-end, explainable deep-learning research application for detecting pneumonia from chest X-ray images.

> [!IMPORTANT]
> **Non-Clinical Software Disclaimer**: MedVision-AI is an educational and AI research demonstration tool. It is **not** a certified medical device and must **not** be used for clinical diagnostic decision-making or patient management.

---

## 📊 Phase 1: RSNA Dataset & Data Engineering Summary

### Official Dataset Attribution
This project uses the official **RSNA Pneumonia Detection Challenge Dataset** provided by the Radiological Society of North America (RSNA), NIH Clinical Center, and Kaggle.
- **Dataset Source:** [Kaggle RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

### Dataset Statistics & Data Quality Audit
- **Total Unique Patients / Images:** `26,684`
- **Class Distribution:**
  - `Normal` / `No Lung Opacity`: `20,672` (77.47%)
  - `Pneumonia` (`Lung Opacity`): `6,012` (22.53%)
- **Data Quality Audit Findings:**
  - `0` duplicate patient records.
  - `0` missing image files.
  - `0` malformed bounding box coordinates ($x \ge 0, y \ge 0, w > 0, h > 0$).
  - 100% target label consistency between CSV manifests.

### Patient Leakage & Group-Aware Split Strategy
To eliminate data leakage, patients are partitioned strictly by `patient_id` using target-stratified group splitting:
- **Train Set (70%):** `18,678` unique patients.
- **Validation Set (15%):** `4,003` unique patients.
- **Test Set (15%):** `4,003` unique patients.
- **Patient Leakage:** **`0.0%` patient overlap** across train, validation, and test splits.

---

## 🗺️ 13-Phase Implementation Roadmap

- [x] **Phase 0: Foundational Hardware-Aware Architecture Setup**
- [x] **Phase 1: Dataset Acquisition & Quality Control EDA** (Current)
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

## 🚀 Quickstart Guide

### 1. Local Environment Setup
```bash
# Clone repository
git clone https://github.com/SwastikPandey1024/MedVision-AI.git
cd MedVision-AI

# Create virtual environment (Python 3.11)
python -m venv .venv
# Activate: Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate

# Install editable package with dev dependencies
pip install -e ".[dev]"
```

### 2. Run Unit Test Suite
```bash
python -m pytest tests/
```
