# 🩺 MedVision-AI: End-to-End Deep Learning System for Chest X-Ray Pneumonia Detection

> **An enterprise-grade, leakage-free computer vision engineering pipeline featuring two-stage transfer learning, rigorous numerical forensics, and validation-isolated threshold optimization on 26,684 chest radiographs.**

---

## 📌 Executive Summary

MedVision-AI is an end-to-end deep learning engineering project designed to classify pneumonia from chest radiographs using the **RSNA Pneumonia Detection Challenge** dataset (26,684 patient studies). Rather than treating model development as a simple training notebook, MedVision-AI implements an industrial-grade ML engineering lifecycle:
- **Zero-Leakage Data Partitioning:** Group-aware stratified splitting across 26,684 unique patients guaranteeing **0% patient leakage** across train (18,678), validation (4,003), and test (4,003) splits.
- **Two-Stage Transfer Learning with BatchNorm Protection:** Feature extraction followed by fine-tuning with strictly frozen Batch Normalization layers to protect pre-trained running statistics from small-batch variance decay.
- **Numerical Forensic Analysis:** Empirical gradient and loss profiling validating numerical stability across full-precision (FP32) and mixed-precision (`mixed_float16`) execution with zero NaN/Inf anomalies.
- **Validation-Only Decision Threshold Optimization:** Optimizing classification threshold strictly on validation predictions, freezing the optimal threshold ($t=0.60$), and applying it to the held-out test split to ensure zero test-data snooping.
- **Scientifically Validated Results:** Achieving **`0.8381 ROC-AUC`** and **`0.6022 PR-AUC`** on the 4,003-patient held-out test set, with **`84.17% Specificity`**, **`79.79% Accuracy`**, and **`64.75% Sensitivity`** at the frozen operating point.

---

## 🎯 Problem Statement & Clinical Context

Pneumonia remains a leading cause of infectious mortality worldwide, accounting for over 2.5 million deaths annually. Rapid and accurate detection on chest X-rays is critical for patient triage, especially in resource-constrained environments.

However, applying deep learning to medical imaging faces major engineering challenges:
1. **Patient Leakage:** Multiple chest X-rays from the same patient across splits causes models to memorize patient-specific anatomical signatures rather than pathology, creating falsely inflated validation metrics.
2. **Class Imbalance:** In screening cohorts, pneumonia cases are typically a minority class (22.5% in the RSNA cohort), rendering raw accuracy misleading and demanding focus on PR-AUC, ROC-AUC, and calibrated operating points.
3. **Fine-Tuning Instability:** Fine-tuning deep convolutional backbones like DenseNet121 often corrupts pre-trained Batch Normalization running statistics, causing gradient divergence or metric collapse.
4. **Evaluation Snooping:** Optimizing decision thresholds directly on test data leads to overly optimistic performance claims that fail to translate into production.

---

## 💡 Engineering Contributions & Solutions

```
+---------------------------------------------------------------------------------------------------+
|                                   MEDVISION-AI ENGINEERING SYSTEM                                 |
+-----------------------------------+-----------------------------------+---------------------------+
| 1. DATA RIGOR                     | 2. TRAINING & FORENSICS           | 3. EVALUATION INTEGRITY   |
| • Stratified Group Splitting      | • Two-stage DenseNet121 pipeline  | • Validation-only search  |
| • 0.0% patient leakage            | • Explicit BatchNorm freeze       | • Zero test snooping      |
| • High-throughput tf.data engine  | • Multi-GPU mixed-precision train | • 4,003-patient test set  |
| • 5% fast dev-loader for CI       | • Empirical numerical forensics   | • ROC-AUC 0.8381, PR 0.60 |
+-----------------------------------+-----------------------------------+---------------------------+
```

### 1. Zero-Leakage Data Engineering
- Ingested 26,684 DICOM/PNG images and associated bounding-box annotations from the RSNA Pneumonia Detection Challenge.
- Implemented target-stratified group k-fold partitioning on `patient_id`, creating an isolated split: **70% Train (18,678 patients)**, **15% Validation (4,003 patients)**, and **15% Held-Out Test (4,003 patients)**.
- Automated unit-tested audits ensure mathematical guarantee of 0% patient intersection across splits.

### 2. Two-Stage Transfer Learning & Architecture
- **Backbone Selection:** DenseNet121 pre-trained on ImageNet, chosen for dense feature reuse and direct gradient propagation across dense blocks.
- **Stage 1 (Feature Extraction):** Backbone weights frozen; custom classification head (GlobalAveragePooling $\to$ BN $\to$ Dropout 0.4 $\to$ Dense 128 ReLU $\to$ Dropout 0.2 $\to$ Dense 1 Sigmoid) trained for 5 epochs with Adam ($LR = 10^{-4}$, clipnorm = 1.0).
- **Stage 2 (Fine-Tuning):** Unfroze top 20 convolutional layers with a low learning rate ($LR = 10^{-5}$). **Critical Design Choice:** Batch Normalization layers remained strictly locked (`trainable = False`) to prevent small-batch noise from corrupting pre-trained channel statistics.

### 3. Numerical Forensic Analysis
- Conducted deep comparative profiling of Stage 2 training under standard 32-bit floating point (FP32) vs. 16-bit mixed precision (`mixed_float16`).
- Confirmed zero exploding gradient events, zero NaN batch occurrences, and exact metric convergence consistency between precision modes.

### 4. Validation-Only Decision Threshold Optimization
- Designed a zero-leakage threshold search engine evaluating 81 candidate thresholds from $0.10$ to $0.90$ ($\Delta t = 0.01$) exclusively on validation set predictions.
- Selected $t=0.60$ (maximizing validation F1-score), froze the threshold, and evaluated the 4,003-patient test split with zero test label visibility during tuning.

---

## 📊 Final Validated Performance Metrics

| Evaluation Split | Total Patients | Decision Threshold | PR-AUC (Primary) | ROC-AUC | F1-Score | Specificity | Sensitivity / Recall | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Split** | 4,003 | 0.50 (Default) | `0.5944` | `0.8358` | `0.5836` | `78.30%` | `71.95%` | `49.09%` | `76.87%` |
| **Validation Split** | 4,003 | 0.60 (Optimal) | `0.5944` | `0.8358` | `0.5898` | `83.84%` | `64.08%` | `54.63%` | `79.39%` |
| **Held-Out Test Split** | 4,003 | 0.50 (Reference) | `0.6023` | `0.8380` | `0.5962` | `78.97%` | `73.17%` | `50.30%` | `77.67%` |
| **Held-Out Test Split** ⭐ | **4,003** | **0.60 (FROZEN)** | **`0.6022`** | **`0.8381`** | **`0.5908`** | **`84.17%`** | **`64.75%`** | **`54.33%`** | **`79.79%`** |

### Confusion Matrix on Held-Out Test Set (4,003 Patients @ Frozen $t = 0.60$):
- **True Positives (TP):** `584` (Correctly identified pneumonia cases)
- **True Negatives (TN):** `2,610` (Correctly identified normal cases)
- **False Positives (FP):** `491` (Normal cases flagged as pneumonia — reduced by 161 compared to $t=0.50$)
- **False Negatives (FN):** `318` (Pneumonia cases missed)

---

## 🛠️ Key Technical & Engineering Decisions

1. **Why DenseNet121?**
   DenseNet's dense connectivity patterns concatenate feature maps from all preceding layers within each dense block. For medical radiographs where subtle interstitial opacities require both low-level edge features and high-level semantic context, DenseNet121 significantly outperforms standard ResNet architectures of comparable parameter count.

2. **Why Freeze BatchNorm During Fine-Tuning?**
   When fine-tuning on domain-specific data with moderate batch sizes, updating Batch Normalization moving statistics frequently degrades feature representations learned on larger datasets. Freezing BN layers preserves calibration and prevents numerical instability.

3. **Why PR-AUC Over Accuracy as Primary Metric?**
   With a 22.5% pneumonia positive rate, a naive majority-class classifier achieves 77.5% accuracy with 0 clinical utility. Precision-Recall AUC (PR-AUC) evaluates model performance directly on the minority positive class across all thresholds, making it the primary indicator of detection capability.

4. **Why Shift Threshold from 0.50 to 0.60?**
   In a triage screening pipeline, high false-positive rates overburden clinical staff with manual review. Adjusting the operating point to $t=0.60$ reduced false positives by 24.7% (from 652 to 491) and boosted specificity to 84.17% while preserving solid sensitivity (64.75%) and invariant global discriminability (0.8381 ROC-AUC).

---

## 💡 Engineering Lessons & Takeaways

- **Data hygiene must precede modeling:** Spending engineering effort up-front on group-aware splitting and data validation prevented premature overfitting and invalid metrics.
- **Threshold selection is an experimental step requiring isolation:** Optimizing thresholds on validation sets rather than test sets is essential to prevent data leakage and ensure realistic expected performance.
- **Fast local iteration unlocks confidence:** Creating a 5% deterministic synthetic/dev data loader enabled running unit tests in seconds on local CPU before dispatching long training jobs to cloud GPUs.

---

## ⚠️ Limitations & Non-Clinical Disclaimer

- **Research Demonstration Only:** MedVision-AI is an academic engineering demonstration. It is not an FDA/EMA-cleared medical device and must **not** be used for clinical diagnosis or treatment decisions.
- **Dataset Constraints:** The RSNA dataset represents adult frontal chest radiographs; performance on pediatric, lateral, or portable emergency room projections has not been evaluated.
- **Binary Scope:** The current production model distinguishes pneumonia from normal/non-opacity cases, but does not differentiate bacterial vs. viral etiologies.

---

## 🔗 Project Links & Artifacts
- **GitHub Repository:** [https://github.com/SwastikPandey1024/MedVision-AI](https://github.com/SwastikPandey1024/MedVision-AI)
- **Architecture Documentation:** [`docs/architecture.md`](docs/architecture.md)
- **Final Metrics Reference:** [`docs/final_metrics.md`](docs/final_metrics.md)
- **Reproducibility Guide:** [`docs/reproducibility.md`](docs/reproducibility.md)
