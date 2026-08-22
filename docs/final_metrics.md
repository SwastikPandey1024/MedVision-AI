# 📊 MedVision-AI: Final Verified Evaluation Metrics

This document consolidates the official, scientifically validated performance metrics for the MedVision-AI DenseNet121 transfer-learning model evaluated on the **RSNA Pneumonia Detection Challenge** dataset.

---

## 🏆 Primary Portfolio Result Summary

> [!IMPORTANT]
> **Evaluation Discipline**: All test evaluation was conducted on a **strictly held-out partition of 4,003 unique patients (0.0% patient leakage)**.
> Decision threshold optimization was conducted **exclusively on validation predictions**. Test labels were **never used during threshold search**.

| Metric Category | Metric Name | Value | Description |
| :--- | :--- | :--- | :--- |
| **Threshold-Independent (Primary)** | **ROC-AUC** | **`0.8381`** | Area Under the Receiver Operating Characteristic Curve |
| **Threshold-Independent (Primary)** | **PR-AUC** | **`0.6022`** | Area Under Precision-Recall Curve (Primary under 22.5% class imbalance) |
| **Operating Point (@ Threshold = 0.60)** | **Accuracy** | **`79.79%`** (`0.7979`) | Overall correct classification rate |
| **Operating Point (@ Threshold = 0.60)** | **Specificity** | **`84.17%`** (`0.8417`) | True negative rate (ruling out normal / non-pneumonia cases) |
| **Operating Point (@ Threshold = 0.60)** | **Sensitivity / Recall** | **`64.75%`** (`0.6475`) | True positive rate (pneumonia case detection) |
| **Operating Point (@ Threshold = 0.60)** | **Precision** | **`54.33%`** (`0.5433`) | Positive predictive value |
| **Operating Point (@ Threshold = 0.60)** | **F1-Score** | **`0.5908`** | Harmonic mean of precision and recall |

---

## 🔬 Detailed Split-by-Split Evaluation Breakdown

### A. Stage 2 Training-Time Validation (Kaggle Multi-GPU Epoch Monitoring)
*Monitored during the 3-epoch fine-tuning phase on 2 × Tesla T4 GPUs with mixed precision (`mixed_float16`):*
- **Validation Loss:** `0.4582`
- **Validation PR-AUC:** `0.6053`
- **Validation ROC-AUC:** `0.8299`

---

### B. Full Validation Split Evaluation (4,003 Unique Patients)
*Full evaluation on the official 15% validation partition used for model selection and threshold search:*
- **Total Patients / Images:** `4,003`
- **Positive Pneumonia Cases:** `902` (22.53%)
- **Negative / Normal Cases:** `3,101` (77.47%)
- **PR-AUC:** `0.5944`
- **ROC-AUC:** `0.8358`
- **Performance @ Threshold 0.50:**
  - **F1-Score:** `0.5836`
  - **Sensitivity / Recall:** `71.95%` (`0.7195`)
  - **Specificity:** `78.30%` (`0.7830`)
  - **Precision:** `49.09%` (`0.4909`)
  - **Accuracy:** `76.87%` (`0.7687`)

---

### C. Full Held-Out Test Split Evaluation @ Default Threshold 0.50 (4,003 Patients)
*Baseline comparison before operating point threshold adjustment:*
- **Total Patients / Images:** `4,003`
- **Positive Cases:** `902` (22.53%)
- **Negative Cases:** `3,101` (77.47%)
- **PR-AUC:** `0.6023`
- **ROC-AUC:** `0.8380`
- **F1-Score:** `0.5962`
- **Sensitivity / Recall:** `73.17%` (`0.7317`)
- **Specificity:** `78.97%` (`0.7897`)
- **Precision:** `50.30%` (`0.5030`)
- **Accuracy:** `77.67%` (`0.7767`)
- **Confusion Matrix Breakdown:**
  - **True Positives (TP):** `660`
  - **True Negatives (TN):** `2,449`
  - **False Positives (FP):** `652`
  - **False Negatives (FN):** `242`

---

### D. Final Held-Out Test Split Evaluation @ Frozen Threshold 0.60 (4,003 Patients)
*Official operating point evaluation using the frozen threshold selected from validation data:*
- **Decision Threshold:** `0.60` (Frozen from validation optimization)
- **PR-AUC:** `0.6022`
- **ROC-AUC:** `0.8381`
- **F1-Score:** `0.5908`
- **Sensitivity / Recall:** `64.75%` (`0.6475`)
- **Specificity:** `84.17%` (`0.8417`)
- **Precision:** `54.33%` (`0.5433`)
- **Accuracy:** `79.79%` (`0.7979`)
- **Confusion Matrix Breakdown:**
  - **True Positives (TP):** `584`
  - **True Negatives (TN):** `2,610`
  - **False Positives (FP):** `491`
  - **False Negatives (FN):** `318`

---

## ⚖️ Threshold Comparison: Default 0.50 vs Frozen 0.60

| Metric | Default Threshold (0.50) | Frozen Optimal Threshold (0.60) | Absolute Delta | Clinical / Engineering Tradeoff |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | `0.8380` | **`0.8381`** | `+0.0001` | Invariant (threshold-independent global separability) |
| **PR-AUC** | `0.6023` | **`0.6022`** | `-0.0001` | Invariant (threshold-independent precision/recall trade-off) |
| **Accuracy** | `77.67%` | **`79.79%`** | **`+2.12%`** | Higher overall correct classification |
| **Specificity** | `78.97%` | **`84.17%`** | **`+5.20%`** | **161 fewer false alarms** (False Positives dropped from 652 to 491) |
| **Precision** | `50.30%` | **`54.33%`** | **`+4.03%`** | Higher confidence in positive flag |
| **Sensitivity** | `73.17%` | **`64.75%`** | `-8.42%` | Expected conservative trade-off at higher confidence threshold |
| **F1-Score** | `0.5962` | **`0.5908`** | `-0.0054` | Balanced harmonic trade-off |

> [!NOTE]
> **Interpretation**: Threshold optimization did not inflate ROC-AUC or PR-AUC (which are threshold-independent). Its engineering purpose was shifting the operating point to reduce false positives by +5.20% specificity and improve precision to 54.33% while maintaining robust ROC-AUC (0.8381).

---

## 🛡️ Validation-Only Decision Threshold Selection Audit

- **Methodology:** Grid search over $t \in [0.10, 0.90]$ with $\Delta t = 0.01$ (81 candidates).
- **Optimization Criterion:** Validation F1-Score.
- **Dataset Partition Used:** Validation split (4,003 samples) **only**.
- **Test Data Participation:** **Zero** (`test_data_used: False`).
- **Audit Artifact:** `artifacts/evaluation/densenet121_stage2_best_threshold_selection_audit.json`
