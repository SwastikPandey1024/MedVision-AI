# 🩺 MedVision-AI: Explainable Chest X-Ray Pneumonia Detection System

[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Deep Learning Framework](https://img.shields.io/badge/TensorFlow-2.16%2B%20%7C%20Keras%203-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Dataset: RSNA](https://img.shields.io/badge/Dataset-RSNA%20Pneumonia%20(26.6k)-blue.svg)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
[![Tests Passing](https://img.shields.io/badge/Tests-74%2F74%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](#-automated-testing--reproducibility)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, leakage-free deep learning system for pneumonia detection from frontal chest radiographs. Built with **DenseNet121** transfer learning, multi-GPU mixed-precision training, numerical stability forensics, and validation-isolated decision threshold optimization.

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
- 🔄 **[Reproducibility Guide](docs/reproducibility.md)** — Exact environment configuration, Kaggle training, and CLI execution.
- 📄 **[Resume Project Bullets](docs/resume_bullets.md)** — ATS-friendly, technical, and one-line resume bullets.
- 💼 **[LinkedIn Descriptions](docs/linkedin_description.md)** — Recruiter-friendly posts and project descriptions.
- 🖼️ **[Demo & Screenshot Catalog](docs/demo_assets.md)** — Screenshot capture plan and visual asset inventory.
- ⚠️ **[Non-Clinical Disclaimer](docs/disclaimer.md)** — Full regulatory and medical scope disclaimer.

---

## 🎯 Problem Statement & Medical Context

Pneumonia accounts for over 2.5 million deaths annually worldwide. While chest radiography (CXR) is the standard initial imaging modality for detection, subtle lung opacities and high clinical workloads create significant triage bottlenecks.

Developing reliable machine learning models for chest X-rays requires overcoming critical engineering challenges:
1. **Patient Memorization (Data Leakage):** Hospital datasets frequently contain multiple radiographs per patient. Splitting datasets at the image level causes models to memorize patient ribcage geometry rather than pathology, creating misleadingly high benchmark numbers.
2. **Class Imbalance:** Pneumonia is typically present in 20–25% of screening images (22.5% in RSNA), making raw accuracy deceptive and requiring PR-AUC and ROC-AUC as primary metrics.
3. **Fine-Tuning Instability:** Fine-tuning deep backbones with small batches often corrupts pre-trained Batch Normalization running statistics, destabilizing loss convergence.
4. **Threshold Snooping:** Tuning classification thresholds directly on test data leads to severe overfitting and inflated claims.

---

## 🏗️ System Architecture & Data Engineering

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
```

### 1. Patient-Aware Group Splitting (0% Leakage)
Patients are partitioned strictly by `patient_id` using target-stratified group k-fold splitting:
- **Train Set (70%):** `18,678` unique patients
- **Validation Set (15%):** `4,003` unique patients
- **Held-Out Test Set (15%):** `4,003` unique patients
- **Leakage Verification:** Automated audit confirms **`0.0%` patient intersection** across all partitions.

### 2. Two-Stage Transfer Learning & BatchNorm Policy
- **Backbone:** DenseNet121 (7,301,185 parameters) pre-trained on ImageNet.
- **Custom Classification Head:** Global Average Pooling $\to$ Batch Normalization $\to$ Dropout ($p=0.4$) $\to$ Dense (128 units, ReLU) $\to$ Dropout ($p=0.2$) $\to$ Dense (1 unit, Sigmoid).
- **Stage 1 (Feature Extraction):** Backbone frozen; classification head trained for 5 epochs with Adam ($LR = 10^{-4}$, gradient clipnorm = 1.0) on 2 × Tesla T4 GPUs with `mixed_float16`.
- **Stage 2 (Fine-Tuning):** Top 20 convolutional layers unfrozen with low learning rate ($LR = 10^{-5}$). **Critical BatchNorm Policy:** All Batch Normalization layers were strictly locked (`trainable = False`) to prevent small-batch noise from corrupting pre-trained channel statistics.

### 3. Numerical Forensics & Precision Profiling
Pre-flight and post-training forensic diagnostics verified:
- Zero NaN/Inf gradients or exploding loss events.
- Exact convergence equivalence between 32-bit floating point (FP32) and 16-bit mixed precision (`mixed_float16`).
- Zero trainable Batch Normalization parameters during Stage 2 fine-tuning.

### 4. Zero-Leakage Decision Threshold Optimization
- Threshold search evaluated 81 candidate thresholds from $0.10$ to $0.90$ ($\Delta t = 0.01$) **exclusively on validation predictions**.
- Optimal operating point selected: **$t = 0.60$** (maximizing validation F1-score).
- Audit trail generated: `test_data_used: False` verified before freezing threshold and evaluating test data.

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

MedVision-AI provides a deterministic 5% synthetic/development data loader enabling complete local execution and CI testing without downloading the full 30 GB dataset.

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/SwastikPandey1024/MedVision-AI.git
cd MedVision-AI

# Create virtual environment (Python 3.11)
python -m venv .venv
.venv\Scripts\activate  # Windows (or: source .venv/bin/activate on Linux/macOS)

# Install in editable mode with development & testing dependencies
pip install -e ".[dev]"
```

### 2. Run Automated Pytest Suite (74 Tests)
```bash
python -m pytest tests/ -q
```

### 3. Local Evaluation Smoke Test (Development Mode)
```bash
python scripts/evaluate.py \
  --checkpoint final_artifacts/densenet121_stage2_best.keras \
  --mode development \
  --split all \
  --optimize-threshold \
  --threshold-criterion f1_score \
  --output-dir artifacts/evaluation
```

### 4. Cloud Multi-GPU Training (Kaggle)
Use the versioned notebook [`notebooks/kaggle/medvision_ai_kaggle_gpu.ipynb`](notebooks/kaggle/medvision_ai_kaggle_gpu.ipynb) on 2 × Tesla T4 GPUs:
```bash
# Stage 1: Feature Extraction
python scripts/train.py --mode full --stage stage1 --epochs 5 --batch-size 64 --mixed-precision --auto-resume

# Stage 2: Fine-Tuning
python scripts/train.py --mode full --stage stage2 --epochs 3 --batch-size 32 --mixed-precision --auto-resume
```

---

## 📁 Repository Structure

```
MedVision-AI/
├── artifacts/                  # Local evaluation metrics, audit JSONs, and plot outputs
│   └── evaluation/             # Confusion matrices, ROC/PR curves, threshold audit reports
├── docs/                       # Project documentation & portfolio assets
│   ├── architecture.md         # System architecture & Mermaid diagrams
│   ├── demo_assets.md          # Visual asset and screenshot catalog
│   ├── disclaimer.md           # Non-clinical research and education disclaimer
│   ├── final_metrics.md        # Comprehensive verified metrics breakdown
│   ├── linkedin_description.md # Recruiter-ready LinkedIn posts and descriptions
│   ├── portfolio.md            # In-depth portfolio case study
│   ├── reproducibility.md      # Step-by-step reproduction guide
│   └── resume_bullets.md       # Tailored resume bullet points
├── notebooks/                  # Interactive experimentation & cloud execution
│   └── kaggle/                 # Multi-GPU Kaggle production training notebook
├── scripts/                    # Command-line entry points
│   ├── evaluate.py             # Multi-split evaluation & threshold optimization engine
│   ├── train.py                # Two-stage training & resume lifecycle runner
│   └── visualize_model.py      # Architecture summary and graph exporter
├── src/medvision/              # Core application package
│   ├── data/                   # DICOM/PNG ingestion, group splitting, tf.data pipeline
│   ├── evaluation/             # Metrics, ROC/PR plotting, threshold search, audit export
│   ├── explainability/         # Grad-CAM saliency mapping (Phase 8)
│   ├── models/                 # Custom CNN baseline & DenseNet121 architecture definitions
│   └── utils/                  # Custom metrics (Specificity, F1), logging, checkpoint tools
└── tests/                      # Automated unit test suite (74 tests)
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
- [ ] **Phase 8: Grad-CAM Saliency & Visual Explainability Engine**
- [ ] **Phase 9: FastAPI / Flask REST API Service**
- [ ] **Phase 10: Streamlit Interactive Radiograph Diagnostic UI**
- [ ] **Phase 11: Docker Containerization & Cloud Deployment**
- [ ] **Phase 12: End-to-End Integration & Load Testing**
- [ ] **Phase 13: Open-Source Community Release & Paper Documentation**

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
