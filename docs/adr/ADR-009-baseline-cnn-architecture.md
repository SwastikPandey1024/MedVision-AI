# ADR-009: Phase 3 Model Architectures, Metrics & Distribution Strategy

* **Status:** Accepted (CTO Approved)
* **Date:** 2026-08-11
* **Authors:** MedVision-AI Engineering Team

---

## Context & Problem Statement

Phase 3 transitions **MedVision-AI** from dataset processing into model architectural design and cloud GPU training. Chest radiograph classification (specifically RSNA Pneumonia Detection) exhibits ~22.5% positive class prevalence, presenting class imbalance. Furthermore, efficient execution requires hybrid deployment (fast CPU dev smoke testing locally and multi-GPU distributed execution on Kaggle Tesla T4 nodes).

---

## Decision Drivers

1. **Class Imbalance Resilience:** Standard accuracy and ROC-AUC can be misleading under 4:1 class imbalance.
2. **Model Diversity:** Need both a simple, fast Custom CNN benchmark and a deep transfer learning model (DenseNet121).
3. **Hardware Utilization:** Multi-GPU scaling (`tf.distribute.MirroredStrategy`) and mixed precision (`mixed_float16`).
4. **Numerical Stability & Fine-Tuning Safety:** Preventing BatchNorm degradation during fine-tuning and preventing FP16 underflow/overflow in sigmoid output heads.

---

## Technical Decisions

### 1. Dual Architectural Hierarchy
* **Custom CNN Baseline (`build_custom_cnn`)**: A 3-block Convolutional-BatchNorm-ReLU-Pooling architecture (32 -> 64 -> 128 filters) with Global Average Pooling and Dropout (0.3/0.5). Designed for fast CPU convergence and lightweight baseline benchmarking.
* **DenseNet121 Primary (`build_densenet121`)**: Pretrained ImageNet backbone with a custom classification head (`Dense(256)` -> `BatchNormalization` -> `Dropout(0.4)` -> `Dense(1)`).
* **Backbone BatchNorm Freeze Rule (`unfreeze_densenet_for_finetuning`)**: During backbone unfreezing, all `BatchNormalization` layers inside the backbone remain explicitly frozen (`trainable = False`) to preserve ImageNet running statistics ($ \mu, \sigma^2 $).

### 2. Primary Metric & Imbalance Handling
* **Checkpointing Metric:** `ModelCheckpoint` and `EarlyStopping` monitor `val_pr_auc` (Precision-Recall Area Under Curve).
* **Class Weighting Calculation:**
  $$W_0 = \frac{N}{2 \times N_0}, \quad W_1 = \frac{N}{2 \times N_1}$$
  Computed strictly from `train_df` labels (never test/validation set).

### 3. Mixed Precision & Multi-GPU Execution
* **Output Head Stability:** The final output layer `Dense(1, activation="sigmoid", dtype="float32")` is locked to `float32` to avoid FP16 overflow/underflow issues.
* **MirroredStrategy:** Dynamically initialized when multiple physical GPUs are present (`len(gpus) > 1`).

---

## Consequences

* **Positive:** Robust PR-AUC optimization, 100% reproducible training, no statistical drift in BatchNorm layers, seamless multi-GPU scaling.
* **Negative:** Slightly higher initial memory footprint during multi-metric evaluation (`PR-AUC`, `ROC-AUC`, `Precision`, `Recall`, `Specificity`, `F1Score`).
