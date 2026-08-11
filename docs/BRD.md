# Business Requirements Document (BRD)

## Project: MedVision-AI
**Document Version:** 1.0  
**Status:** Baseline Established  

---

## 1. Business Objectives & Strategic Context
AI in medical diagnostics is growing rapidly, but adoption remains hindered by lack of explainability, data leakage during model validation, and monolithic software codebases. MedVision-AI demonstrates how to build a modular, reproducible, explainable, and compliant deep-learning research architecture.

### Primary Objectives
- **Standardize Research Workflows:** Provide a template for medical vision projects using TensorFlow/Keras 3.
- **Mitigate Methodological Risk:** Enforce patient-level splitting to prevent data leakage across train/val/test splits.
- **Demonstrate Responsible AI:** Integrate spatial visual explainability (Grad-CAM) to foster trust and transparency.

---

## 2. Regulatory, Ethical & Privacy Compliance

> [!CAUTION]
> **Regulatory Disclaimer:** MedVision-AI is explicitly **not** intended for clinical use and has **not** received FDA 510(k), CE mark, or Software as a Medical Device (SaMD) clearance.

### Data Privacy Controls
- All datasets used (e.g. RSNA Pneumonia Challenge) must be fully anonymized open-access research datasets.
- No protected health information (PHI) or patient names shall be stored or transmitted.
- Datasets are ignored by Git (`.gitignore`) to guarantee zero accidental exposure of raw files.

---

## 3. Risk Assessment & Mitigation Matrix

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Data Leakage (Same patient in train & test)** | High | Implement mandatory `PatientID`-grouped splitters (`src/medvision/data/splits.py`). |
| **Model Over-optimism on Single Metric** | Medium | Evaluate using ROC-AUC, PR-AUC, Sensitivity, Specificity, and Confusion Matrix rather than Accuracy alone. |
| **Black Box AI Confusion** | Medium | Overlay Grad-CAM heatmaps on inference outputs to display spatial feature focus. |
| **Dependency Version Drift** | Low | Pin dependency versions in `pyproject.toml` and `requirements.txt`. |
