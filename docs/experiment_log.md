# 🧪 MedVision-AI: Empirical Experiment & Training Log

> **Chronological record of model training runs, hardware configurations, and evaluation audits.**

---

## 🏃 Experiment Run History

### Experiment 01: Baseline Custom CNN
- **Architecture:** 3-block Conv2D + BatchNorm + MaxPool + Dense Head (328k params)
- **Dataset:** 5% synthetic/development subset
- **Objective:** Verify pipeline execution, loss reduction, and metrics calculation.
- **Outcome:** Successful convergence; validated loss reduction and custom Specificity/F1 metric tracking.

---

### Experiment 02: DenseNet121 Stage 1 (Feature Extraction)
- **Hardware:** 2 × NVIDIA Tesla T4 GPUs (Kaggle)
- **Strategy:** `tf.distribute.MirroredStrategy` + `mixed_float16`
- **Hyperparameters:** Batch size = 64, Epochs = 5, Optimizer = Adam ($LR = 10^{-4}$, clipnorm = 1.0)
- **Backbone Status:** Frozen (`trainable = False`)
- **Outcome:** Loss converged from 0.5821 to 0.4712; Validation PR-AUC reached 0.5812. Checkpoint saved: `final_artifacts/densenet121_stage1_best.keras`.

---

### Experiment 03: DenseNet121 Stage 2 (Controlled Fine-Tuning)
- **Hardware:** 2 × NVIDIA Tesla T4 GPUs
- **Hyperparameters:** Batch size = 32, Epochs = 3, Optimizer = Adam ($LR = 10^{-5}$, clipnorm = 1.0)
- **Backbone Status:** Top 20 conv layers unfrozen; **All BatchNorm layers FROZEN (`trainable = False`)**
- **Forensic Checks:** Verified 0 NaN/Inf gradients, 0 trainable BatchNorm parameters.
- **Outcome:** Validation PR-AUC improved to 0.6053, Validation ROC-AUC to 0.8299. Checkpoint saved: `final_artifacts/densenet121_stage2_best.keras`.

---

### Experiment 04: Zero-Leakage Validation Threshold Optimization
- **Data Used:** Validation Split only (4,003 unique patients)
- **Grid:** $t \in [0.10, 0.90]$, step = $0.01$ (81 evaluations)
- **Criterion:** Maximize Validation F1-Score
- **Optimal Threshold:** **$t = 0.60$** (Validation F1 = 0.5898, Specificity = 83.84%)
- **Audit File:** `artifacts/evaluation/densenet121_stage2_best_threshold_selection_audit.json` (`test_data_used: False`)

---

### Experiment 05: Official Held-Out Test Evaluation
- **Data Used:** Held-Out Test Split (4,003 unique patients, 0.0% patient leakage)
- **Frozen Threshold Applied:** **$0.60$**
- **Verified Primary Results:**
  - **ROC-AUC:** `0.8381`
  - **PR-AUC:** `0.6022`
  - **Specificity:** `84.17%` (2,610 / 3,101 TN)
  - **Accuracy:** `79.79%`
  - **Sensitivity:** `64.75%` (584 / 902 TP)
  - **Precision:** `54.33%`
  - **F1-Score:** `0.5908`
- **Confusion Matrix:** TP=584, TN=2610, FP=491, FN=318.
