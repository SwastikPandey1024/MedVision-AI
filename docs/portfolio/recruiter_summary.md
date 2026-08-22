# 💼 MedVision-AI: Recruiter & Executive Summary

> **Concise, impact-driven summary tailored for technical recruiters, engineering directors, and talent acquisition teams.**

---

## 🎯 Candidate Project Overview: MedVision-AI

| Criteria | Candidate Delivery / Project Fact |
| :--- | :--- |
| **Domain** | Medical Computer Vision & Machine Learning Engineering |
| **Model** | DenseNet121 (7.3M parameters, Two-Stage Transfer Learning) |
| **Dataset** | RSNA Pneumonia Detection Challenge (26,684 Chest Radiographs) |
| **Primary Achievements** | **0.8381 ROC-AUC**, **0.6022 PR-AUC**, **84.17% Specificity** on 4,003 Unseen Patients |
| **Data Discipline** | **0.0% Patient Data Leakage** via Stratified Group Splitting on `patient_id` |
| **Evaluation Rigor** | Decision threshold ($t=0.60$) tuned strictly on validation set; zero test snooping |
| **Explainability** | Custom **Grad-CAM** saliency mapping highlighting lung opacities |
| **Full Stack** | **FastAPI** REST API, **Streamlit** Interactive UI, **Docker** containerization |
| **Test Quality** | **92 / 92 unit and integration tests passing (100%)** |
| **Code Quality** | Clean modular architecture, typed Python 3.11, structured logging, full documentation |

---

## 💡 Top Skills Demonstrated
- **Deep Learning Frameworks:** TensorFlow 2.16+, Keras 3, DenseNet121, Transfer Learning, Mixed Precision.
- **MLOps & Evaluation:** Group-Aware Splitting, Data Leakage Audits, Calibration, ROC/PR Curve Analysis, Precision-Recall Optimization.
- **Software Engineering:** FastAPI, Streamlit, Pytest, Docker, Pydantic, DICOM parsing (`pydicom`), REST APIs.
- **Explainable AI (XAI):** Grad-CAM, GradientTape Saliency, Colormap Blending.
