# 🏗️ MedVision-AI: System Architecture & End-to-End Pipeline

This document details the software architecture, data processing topology, staged model training lifecycle, and evaluation engine of MedVision-AI.

---

## 🗺️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph DataEngineering ["1. Data Engineering & Group Splitting"]
        RSNA["RSNA Pneumonia Dataset<br/>(26,684 DICOM/PNG Images)"] --> Manifest["Manifest Parsing & QA Audit<br/>(Bounding Box & Label Validation)"]
        Manifest --> StratifiedGroup["Target-Stratified Group Splitting<br/>(Partitioned strictly by patient_id)"]
        StratifiedGroup --> TrainSet["Train Split (70%)<br/>18,678 Patients (0% Leakage)"]
        StratifiedGroup --> ValSet["Validation Split (15%)<br/>4,003 Patients (0% Leakage)"]
        StratifiedGroup --> TestSet["Held-Out Test Split (15%)<br/>4,003 Patients (0% Leakage)"]
    end

    subgraph PipelineEngine ["2. Input Processing & tf.data Engine"]
        TrainSet --> Augment["Augmentation Pipeline<br/>(Rotate, Zoom, Contrast, Flip)"]
        Augment --> TFDataTrain["tf.data Pipeline (Prefetched, Batched)"]
        ValSet --> TFDataVal["Validation tf.data Pipeline (Deterministic)"]
        TestSet --> TFDataTest["Test tf.data Pipeline (Deterministic)"]
    end

    subgraph TrainingLifecycle ["3. Two-Stage Model Training Lifecycle"]
        TFDataTrain --> Stage1["Stage 1: Feature Extraction<br/>(DenseNet121 Backbone Frozen, LR=1e-4, Adam)"]
        Stage1 --> Ckpt1["Stage 1 Best Checkpoint<br/>(PR-AUC & ROC-AUC Monitored)"]
        Ckpt1 --> Stage2["Stage 2: Fine-Tuning<br/>(Top 20 Layers Unfrozen, BatchNorm FROZEN, LR=1e-5)"]
        Stage2 --> Forensics["Numerical Forensic Analysis<br/>(FP32 vs MP Check, 0 Bad Batches, 0 Trainable BN)"]
        Forensics --> Ckpt2["Stage 2 Best Production Model<br/>(7,301,185 Parameters)"]
    end

    subgraph EvaluationEngine ["4. Zero-Leakage Evaluation & Reporting Engine"]
        Ckpt2 --> ValEval["Validation Set Evaluation<br/>(4,003 Patients, Single-Pass Inference)"]
        TFDataVal --> ValEval
        ValEval --> ThreshOpt["Validation-Only Threshold Search<br/>(81 Candidates: 0.10 to 0.90, F1 Criterion)"]
        ThreshOpt --> FrozenThresh["Frozen Decision Threshold: 0.60<br/>(Audit Saved: test_data_used = False)"]
        FrozenThresh --> TestEval["Held-Out Test Evaluation<br/>(4,003 Patients @ Threshold 0.60)"]
        TFDataTest --> TestEval
        TestEval --> Artifacts["Evaluation Artifacts<br/>(JSON Reports, Markdown Summaries, ROC/PR Curves, Confusion Matrix)"]
    end

    style RSNA fill:#2b3a4a,stroke:#4a90e2,stroke-width:2px,color:#fff
    style StratifiedGroup fill:#1e3d2f,stroke:#27ae60,stroke-width:2px,color:#fff
    style Stage2 fill:#4a2b3a,stroke:#e74c3c,stroke-width:2px,color:#fff
    style Forensics fill:#3d3a1e,stroke:#f39c12,stroke-width:2px,color:#fff
    style ThreshOpt fill:#2c3e50,stroke:#3498db,stroke-width:2px,color:#fff
    style FrozenThresh fill:#1b4f72,stroke:#5dade2,stroke-width:2px,color:#fff
    style Artifacts fill:#1e3d2f,stroke:#2ecc71,stroke-width:2px,color:#fff
```

---

## 🧩 Architectural Modules Breakdown

### 1. Data Engineering & Group Splitting (`medvision.data`)
- **RSNA Manifest Parsing:** Ingests CSV metadata, cross-verifies bounding boxes ($x, y, w, h$), and maps multi-box records to patient-level binary classification targets (`0` = Normal/No Opacity, `1` = Pneumonia).
- **Patient-Aware Splitter:** Implements target-stratified group k-fold partitioning on `patient_id`. Audited automatically to enforce exactly **0% patient leakage** across train, validation, and test sets.
- **Development Loader:** Provides a 5% deterministic subset loader for lightweight local CPU unit testing and rapid CI verification without requiring the full 30 GB dataset.

### 2. High-Performance Input Pipeline (`medvision.data.pipeline`)
- **`tf.data.Dataset` Architecture:** Implements parallel decoding, image normalization, spatial resizing to $(224 \times 224 \times 3)$, and memory prefetching (`AUTOTUNE`).
- **Data Augmentation Engine:** Implements subtle medical-imaging-safe spatial transforms (random rotation $\pm 7^\circ$, random zoom $\pm 5\%$, horizontal flipping, contrast jitter $\pm 10\%$) applied exclusively during training.

### 3. Model Training & Fine-Tuning Lifecycle (`medvision.models`)
- **DenseNet121 Architecture:** Pre-trained on ImageNet with customized classification head: Global Average Pooling $\to$ Batch Normalization $\to$ Dropout ($p=0.4$) $\to$ Dense (128 units, ReLU, L2 regularized) $\to$ Dropout ($p=0.2$) $\to$ Dense (1 unit, Sigmoid).
- **Stage 1 (Feature Extraction):** Backbone frozen (`trainable = False`), classifier head trained with Adam ($LR = 10^{-4}$, gradient clipnorm = 1.0).
- **Stage 2 (Controlled Fine-Tuning):** Top 20 convolutional layers unfreezed. **Batch Normalization layers are explicitly locked (`trainable = False`)** to prevent non-i.i.d. running mean/variance corruption. Trained with low learning rate ($LR = 10^{-5}$).
- **Numerical Forensics:** Automated preflight verifies zero NaN/Inf gradients, zero exploding loss spikes, and exact FP32/mixed-precision equivalence.

### 4. Zero-Leakage Evaluation & Audit Engine (`medvision.evaluation`)
- **Strict Partition Isolation:** Threshold optimization receives only validation predictions (`val_y_true`, `val_y_pred_prob`).
- **Audit Persistence:** Generates `{model}_threshold_selection_audit.json` and `.md` recording the full 81-candidate scan history and verifying `test_data_used == False`.
- **Operating Point Evaluation:** The frozen threshold is applied to the 4,003-patient test split to generate multi-metric reports, confusion matrices, ROC curves, and Precision-Recall curves.
