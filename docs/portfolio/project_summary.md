# 🩺 MedVision-AI: Executive Project Summary

> **An enterprise-grade, leakage-free deep learning system for chest radiograph pneumonia detection featuring two-stage transfer learning, numerical forensics, validation-isolated threshold optimization, Grad-CAM visual explainability, and full REST/UI deployment.**

---

## 🌟 High-Impact Highlights

- **0.0% Patient Data Leakage:** Stratified group k-fold partitioning on `patient_id` across 26,684 chest radiographs mathematically guarantees zero patient overlap across 18,678 train, 4,003 validation, and 4,003 held-out test records.
- **Two-Stage DenseNet121 Transfer Learning:** Feature extraction followed by fine-tuning with strictly locked Batch Normalization layers to preserve pre-trained ImageNet channel statistics.
- **Empirical Numerical Forensics:** Gradient and loss profiling verifying zero NaN/Inf events, absence of initial-batch collapse, and exact convergence equivalence across FP32 and `mixed_float16`.
- **Validation-Only Decision Threshold Optimization:** Operating threshold ($t = 0.60$) tuned exclusively on validation predictions with zero test snooping, reducing false alarms by **24.7% (161 fewer false positives)** on unseen test data.
- **Scientifically Validated Held-Out Results (4,003 Patients):**
  - **ROC-AUC (Primary):** **`0.8381`**
  - **PR-AUC (Primary):** **`0.6022`**
  - **Specificity:** **`84.17%`**
  - **Accuracy:** **`79.79%`**
  - **Sensitivity:** **`64.75%`**
  - **Precision:** **`54.33%`**
  - **F1-Score:** **`0.5908`**
- **Production-Ready Engineering System:**
  - Post-hoc **Grad-CAM** saliency maps highlighting lung parenchymal opacities.
  - High-performance asynchronous **FastAPI** REST API.
  - Interactive **Streamlit** radiologist dashboard with real-time threshold slider and DICOM windowing.
  - Slim non-root **Docker** containerization with health checks.
  - 100% passing automated test suite (92 tests passing).
