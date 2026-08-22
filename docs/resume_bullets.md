# 📄 MedVision-AI: Resume Project Bullets

Use these resume bullets to showcase MedVision-AI on technical software engineering, machine learning engineering, or data science resumes.

---

## Version A: ATS-Friendly & Concise (Recommended for General Software / ML Engineering)

**MedVision-AI — Explainable Chest X-Ray Deep Learning System** | *Python, TensorFlow/Keras, NumPy, Scikit-Learn*
- Developed an end-to-end medical computer vision system classifying pneumonia on 26,684 chest X-rays, achieving **0.8381 ROC-AUC** and **0.6022 PR-AUC** on a held-out test set of 4,003 patients.
- Engineered a target-stratified group partitioning pipeline enforcing **0% patient data leakage** across 70/15/15 train, validation, and test splits across 26,684 patient records.
- Implemented a two-stage DenseNet121 transfer learning architecture with frozen Batch Normalization layers during fine-tuning to prevent feature degradation.
- Designed a validation-isolated threshold optimization engine (81 candidate thresholds), freezing an optimal decision threshold ($t=0.60$) that achieved **84.17% specificity** and **79.79% accuracy** on unseen test data.
- Built automated test suites and numerical forensic checks ensuring zero NaN/Inf gradient events across full-precision (FP32) and 16-bit mixed-precision (`mixed_float16`) execution.

---

## Version B: Technical & Deep ML Engineering (Recommended for Senior ML / AI Research Roles)

**MedVision-AI — Production-Grade Medical Imaging Pipeline** | *Python 3.11, TensorFlow 2.16+, Keras 3, tf.data, Pytest*
- Architected a scalable deep learning pipeline on the RSNA Pneumonia Detection Challenge dataset (26,684 studies), resolving minority-class imbalance (22.5% positive rate) with PR-AUC-guided optimization.
- Prevented patient identity memorization by engineering a group-aware `patient_id` splitting engine with automated audit guarantees of **0.0% patient leakage**.
- Orchestrated two-stage DenseNet121 training on 2 × Tesla T4 GPUs using `tf.distribute.MirroredStrategy` and mixed precision, locking BatchNorm moving statistics to safeguard pre-trained representations.
- Executed numerical gradient profiling and forensic checks confirming numerical stability, absence of first-batch gradient collapse, and identical loss trajectory across FP32 and FP16 modes.
- Enforced strict zero-snooping evaluation by searching decision thresholds exclusively on validation predictions ($t=0.60$), yielding **0.8381 ROC-AUC**, **0.6022 PR-AUC**, **84.17% specificity**, and **161 fewer false positives** on the 4,003-patient test split.

---

## Version C: High-Impact One-Line Summaries (For Compact / Single-Page Resumes)

- **Option 1 (Results-Driven):** Built an end-to-end DenseNet121 chest X-ray classification system achieving **0.8381 ROC-AUC** and **84.17% specificity** on 4,003 held-out patients with zero patient leakage.
- **Option 2 (Engineering-Driven):** Engineered a leakage-free medical imaging pipeline on 26,684 chest radiographs featuring two-stage DenseNet121 training, numerical forensics, and validation-only threshold selection (**0.8381 ROC-AUC**).
- **Option 3 (Systems-Driven):** Designed a high-throughput medical computer vision system with group-stratified splitting, mixed-precision distributed training, and automated threshold audits (**79.79% accuracy, 0.8381 ROC-AUC**).
