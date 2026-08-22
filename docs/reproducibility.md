# 🔄 MedVision-AI: Complete Reproducibility & Execution Guide

This document provides step-by-step instructions to replicate the data splitting, local testing, GPU training, and zero-leakage evaluation pipelines of MedVision-AI.

---

## 💻 1. Environment & Prerequisites

### System Requirements
- **Python Version:** `3.11.x` (Tested with Python 3.11.9)
- **Deep Learning Framework:** TensorFlow `2.16.x` / `2.20.x` with Keras `3.x`
- **Operating Systems:** Windows 10/11, Ubuntu 22.04+, or macOS
- **Hardware Acceleration:**
  - *Local Development / Testing:* Standard Multi-core CPU (uses built-in 5% synthetic/development data loader).
  - *Full Model Training:* CUDA-compatible GPU (e.g., 2 × NVIDIA Tesla T4 or single RTX 3080/4090 with $\ge 16\text{ GB}$ VRAM).

---

## 📦 2. Local Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/SwastikPandey1024/MedVision-AI.git
cd MedVision-AI

# 2. Create isolated Python 3.11 virtual environment
python -m venv .venv

# 3. Activate environment
# On Windows (PowerShell / Command Prompt):
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 4. Install editable package with all development & evaluation dependencies
pip install -e ".[dev]"
```

---

## 🧪 3. Running the Automated Test Suite

Verify that all 74 unit tests across data engineering, splitting, architecture, metrics, and threshold audits pass:

```bash
# Run full pytest test suite
python -m pytest tests/ -q
```

*Expected Output:*
```
.......................................................................... [100%]
74 passed in ~45s
```

---

## ⚡ 4. Lightweight Development Evaluation (Local CPU)

To test the evaluation engine, threshold search, and metric reporting locally without downloading the full 30 GB dataset:

```bash
# Run multi-split evaluation in development mode with validation threshold optimization
python scripts/evaluate.py \
  --checkpoint final_artifacts/densenet121_stage2_best.keras \
  --mode development \
  --split all \
  --optimize-threshold \
  --threshold-criterion f1_score \
  --batch-size 8 \
  --output-dir artifacts/evaluation
```

*Expected Outputs:*
- `artifacts/evaluation/densenet121_stage2_best_threshold_selection_audit.json`
- `artifacts/evaluation/densenet121_stage2_best_threshold_selection_audit.md`
- `artifacts/evaluation/densenet121_stage2_best_val_report.json` & `.md`
- `artifacts/evaluation/densenet121_stage2_best_test_report.json` & `.md`
- `artifacts/evaluation/model_comparison_report.md`

---

## 🚀 5. Production Multi-GPU Training (Kaggle / Cloud GPU)

For full-scale training on the complete 26,684-image RSNA dataset:

1. Open the versioned Kaggle notebook: [`notebooks/kaggle/medvision_ai_kaggle_gpu.ipynb`](../notebooks/kaggle/medvision_ai_kaggle_gpu.ipynb).
2. Attach the official dataset: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).
3. Enable GPU Accelerator (`2 × NVIDIA Tesla T4` or `1 × P100`).
4. Execute Stage 1 feature extraction:
   ```bash
   python scripts/train.py \
     --mode full \
     --stage stage1 \
     --epochs 5 \
     --batch-size 64 \
     --mixed-precision \
     --auto-resume
   ```
5. Execute Stage 2 controlled fine-tuning:
   ```bash
   python scripts/train.py \
     --mode full \
     --stage stage2 \
     --epochs 3 \
     --batch-size 32 \
     --mixed-precision \
     --auto-resume
   ```

---

## 📊 6. Full Patient-Held-Out Evaluation

Once Stage 2 training completes, execute the official evaluation:

```bash
# Evaluate full 4,003-patient validation & held-out test splits with frozen threshold selection
python scripts/evaluate.py \
  --checkpoint models/checkpoints/densenet121_stage2_best.keras \
  --mode full \
  --split all \
  --optimize-threshold \
  --threshold-criterion f1_score \
  --batch-size 32 \
  --output-dir artifacts/evaluation
```

---

## 🗄️ 7. Artifact & Checkpoint Hygiene Policy

- **Version Control:** Git tracks code, configs, notebooks, tests, and documentation. Large binary `.keras` checkpoints and raw DICOM/PNG datasets are strictly ignored via `.gitignore`.
- **Session-Local Artifacts:** In cloud environments (e.g., Kaggle `/kaggle/working/`), download or backup trained checkpoints (`densenet121_stage1_best.keras`, `densenet121_stage2_best.keras`) and metric summaries to local `final_artifacts/` before terminating GPU runtimes.
