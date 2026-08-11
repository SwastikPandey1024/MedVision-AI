# Product Requirements Document (PRD)

## Project: MedVision-AI
**Document Version:** 1.0  
**Status:** Approved (Phase 0 Foundational Baseline)  
**Target Domain:** Medical Image Analysis / Deep Learning Research & Education  

---

## 1. Product Overview & Vision
MedVision-AI is an end-to-end explainable artificial intelligence system designed to assist medical researchers, machine learning students, and developers in exploring deep learning techniques applied to chest X-ray image analysis.

### Core Value Proposition
- **Explainable Classification:** Combines high-accuracy DenseNet121 transfer learning with spatial Grad-CAM heatmaps so predictions are interpretable rather than a "black box".
- **Enterprise Engineering:** Built using modern Python packaging, decoupled microservice design (Flask + Streamlit), and containerization.
- **Data Integrity:** Strict patient-aware data splitting rules to prevent data leakage and benchmark over-optimism.

---

## 2. Target Audience & User Personas

| Persona | Role | Primary Need |
| :--- | :--- | :--- |
| **Dr. Elena Vance** | Clinical AI Researcher | Wants an open-source framework to test visual explainability against RSNA chest X-rays without building code from scratch. |
| **Marcus Chen** | Machine Learning Student | Wants a production-grade template demonstrating transfer learning, `tf.data`, unit tests, and API deployment. |
| **DevOps Developer** | Deployment Engineer | Requires a clean REST API interface with containerized Docker deployment options. |

---

## 3. Scope & Non-Goals

### In-Scope (Phase 0–13)
- Chest X-ray binary classification (Normal vs Pneumonia).
- DenseNet121 feature extraction and fine-tuning.
- Grad-CAM heatmap visualization.
- Flask REST API (`/health`, `/predict`).
- Streamlit interactive web user interface.
- Dockerized deployment setup.

### Out-of-Scope (Non-Goals)
- Multi-label classification (e.g. Cardiomegaly, Atelectasis, Effusion simultaneously).
- Direct DICOM PACs hospital software integration.
- Clinical diagnostic automated decision-making.

---

## 4. Success Criteria & Key Performance Indicators (KPIs)

1. **Model Performance:** ROC-AUC $\ge 0.90$, Sensitivity/Recall $\ge 0.88$ on patient-held-out test set.
2. **API Latency:** Flask `/predict` response time $< 1.5\text{ seconds}$ per image inference on standard CPU.
3. **Interpretability Quality:** Grad-CAM heatmaps correctly target pulmonary opacity regions on positive X-ray samples.
4. **Code Quality:** $\ge 80\%$ unit test code coverage and zero severe linter violations.
