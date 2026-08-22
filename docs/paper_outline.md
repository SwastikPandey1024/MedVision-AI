# 📄 MedVision-AI: Technical Paper & Research Manuscript Outline

> **Structured manuscript draft for academic or industrial publication: "MedVision-AI: An End-to-End Leakage-Free Chest Radiograph Classification and Explainability Pipeline".**

---

## 1. Abstract
Chest radiography is the frontline diagnostic imaging modality for respiratory disease, yet automated computer vision systems frequently suffer from silent patient leakage, fine-tuning divergence, and over-optimistic evaluation snooping. We present **MedVision-AI**, an open-source, leakage-free deep learning system for pneumonia detection evaluated on 26,684 radiographs from the RSNA Pneumonia Detection Challenge. By implementing target-stratified group k-fold partitioning by patient ID, we guarantee 0% patient leakage. Using a two-stage transfer learning regime with DenseNet121 and locked Batch Normalization layers, combined with validation-isolated decision threshold optimization ($t=0.60$), the model achieves **`0.8381 ROC-AUC`**, **`0.6022 PR-AUC`**, and **`84.17% Specificity`** on a strictly held-out test cohort of 4,003 unique patients. A post-hoc Grad-CAM saliency engine provides visual interpretability over focal lung opacities, exposed via a FastAPI REST service and interactive Streamlit UI.

---

## 2. Introduction & Clinical Motivation
- Global burden of pneumonia (>2.5 million deaths/year).
- Radiologist workload, diagnostic delay, and emergency triage bottlenecks.
- Engineering pitfalls in published medical AI literature (patient leakage, threshold snooping, uncalibrated metrics).

---

## 3. Dataset & Data Engineering
- **RSNA Pneumonia Detection Challenge Dataset:** 26,684 frontal radiographs.
- **Group-Aware Splitting Methodology:** Partitioning on `patient_id` (70% Train: 18,678 patients, 15% Val: 4,003 patients, 15% Test: 4,003 patients).
- **Leakage Verification Audit:** Mathematical guarantee of zero patient overlap across all sets.
- **DICOM Processing & CR/DX VOI LUT Windowing.**

---

## 4. Model Architecture & Staged Training Lifecycle
- **Backbone Selection:** DenseNet121 (7,301,185 parameters).
- **Classification Head:** GlobalAveragePooling $\to$ BN $\to$ Dropout (0.4) $\to$ Dense (128, ReLU) $\to$ Dropout (0.2) $\to$ Dense (1, Sigmoid).
- **Stage 1 (Feature Extraction):** Backbone frozen; trained with Adam ($LR=10^{-4}$, clipnorm=1.0) with mixed precision (`mixed_float16`).
- **Stage 2 (Controlled Fine-Tuning):** Top 20 conv layers unfrozen ($LR=10^{-5}$); **BatchNorm layers strictly locked (`trainable=False`)**.
- **Numerical Forensics:** Stability profiling between FP32 and mixed-precision.

---

## 5. Zero-Snooping Evaluation & Threshold Selection
- Isolation of validation split for operating point search.
- 81-candidate grid scan ($t \in [0.10, 0.90]$).
- Selection of $t=0.60$ maximizing validation F1.
- Application to frozen test cohort ($N=4,003$ patients).

---

## 6. Results & Benchmark Comparison
- **Validation Split ($N=4,003$):** ROC-AUC `0.8358`, PR-AUC `0.5944`, F1 `0.5898`, Specificity `83.84%` @ $t=0.60$.
- **Held-Out Test Split ($N=4,003$):** ROC-AUC `0.8381`, PR-AUC `0.6022`, F1 `0.5908`, Specificity `84.17%`, Sensitivity `64.75%`, Accuracy `79.79%` @ $t=0.60$.
- **Operating Trade-Off:** 161 fewer false alarms compared to default 0.50 threshold.

---

## 7. Explainability, API, and Deployment
- Grad-CAM on layer `conv5_block16_2_conv`.
- FastAPI asynchronous REST API service with Base64 visual responses.
- Interactive Streamlit dashboard.
- Containerization using slim non-root Docker images.

---

## 8. Limitations & Non-Clinical Disclaimer
- Adult frontal projections only; lateral and pediatric views excluded.
- Binary detection scope (opacity presence vs. etiology differentiation).
- Research prototype status (not FDA/CE cleared).

---

## 9. Conclusion & Open Science
- Availability of open-source codebase, audit reports, and container configs on GitHub.
