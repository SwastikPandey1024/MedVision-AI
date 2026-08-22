# 🎙️ MedVision-AI: Presentation & Interview Demo Scripts

> **Structured verbal narratives for portfolio walkthroughs, technical interviews, and hiring manager presentations.**

---

## ⚡ 1. 30-Second Elevator Pitch (Recruiter / Overview)

> "MedVision-AI is an enterprise-grade medical computer vision system for detecting pneumonia from chest X-rays using DenseNet121 on the RSNA dataset. Unlike naive classifiers that suffer from silent patient leakage and data snooping, MedVision-AI implements 0% patient leakage by partitioning on patient ID, two-stage transfer learning with frozen BatchNorm protection, and validation-only threshold optimization ($t=0.60$). It achieves **0.8381 ROC-AUC** and **84.17% specificity** on a strictly held-out test cohort of 4,003 unique patients, and is fully packaged with a Grad-CAM explainability engine, FastAPI backend, and interactive Streamlit UI."

---

## ⏱️ 2. 2-Minute Technical Deep-Dive (Hiring Manager / Team Lead)

> "When building computer vision models for healthcare, the primary engineering challenge is rarely the depth of the neural network — it’s rigorous data hygiene and evaluation discipline.
>
> In MedVision-AI, hospital patients often have multiple sequential radiographs. Splitting at the image level leaks patient ribcage anatomy across sets. I engineered a target-stratified group k-fold partitioning pipeline on `patient_id` across 26,684 records, guaranteeing **0.0% patient leakage** across 70/15/15 train, validation, and test splits.
>
> For modeling, I implemented a two-stage DenseNet121 pipeline. In Stage 1, we froze the backbone and trained a custom regularized classification head with Adam and mixed precision. In Stage 2, we selectively unfroze the top 20 convolutional layers while **strictly locking Batch Normalization layers** to prevent small-batch noise from destroying ImageNet channel statistics.
>
> To avoid test snooping, I designed an 81-point grid search executed exclusively on validation predictions to select an optimal operating point ($t=0.60$). On our 4,003-patient held-out test set, this delivered **0.8381 ROC-AUC**, **0.6022 PR-AUC**, and **84.17% specificity** — cutting false alarms by 161 cases compared to default thresholds.
>
> Finally, I productionized the model with a Grad-CAM visual explainability engine targeting `conv5_block16_2_conv`, an async FastAPI REST API, an interactive Streamlit diagnostic dashboard with DICOM VOI LUT support, and a non-root Docker deployment."

---

## 🎯 3. 5-Minute Senior ML / Architecture Interview Narrative

### Section 1: Problem & System Design
> "In medical imaging ML, naive workflows fail because of two hidden bugs: patient identity leakage and Batch Normalization degradation. When I built MedVision-AI on the RSNA dataset of 26,684 chest X-rays, my first step was auditing the metadata. Because patients frequently have multiple scans, random splitting causes the model to memorize patient-specific bone structure instead of lung opacities. I implemented group-aware stratified splitting on `patient_id`, creating an uncontaminated 4,003-patient test split with automated unit test verification."

### Section 2: Transfer Learning & Numerical Forensics
> "For our backbone, I chose DenseNet121 because its dense connections concatenate multi-scale feature maps from earlier layers, preserving low-level textural detail alongside high-level semantic context. 
> 
> During Stage 2 fine-tuning, I enforced a strict rule: all Batch Normalization layers remained non-trainable. In moderate-batch domain fine-tuning, allowing BN moving mean and variance to update causes feature drift and divergence. We also ran empirical gradient profiling between FP32 and `mixed_float16` to guarantee zero NaN/Inf occurrences or loss explosions."

### Section 3: Zero-Leakage Threshold Optimization & Metrics
> "Because pneumonia represents a 22.5% minority class, raw accuracy is deceptive. I treated PR-AUC and ROC-AUC as primary invariant metrics. Furthermore, rather than using an arbitrary 0.50 threshold, I ran an 81-point sweep exclusively across validation set predictions to optimize the F1-score. Freezing the resulting $t=0.60$ threshold and evaluating on the 4,003 unseen test patients yielded **0.8381 ROC-AUC**, **0.6022 PR-AUC**, **79.79% accuracy**, and **84.17% specificity**, saving 161 false positives."

### Section 4: Explainability, Serving & Deployment
> "To deliver transparency, I implemented Grad-CAM on `conv5_block16_2_conv` using TensorFlow's `GradientTape`, generating normalized saliency overlays over focal opacities without mutating model weights. The system is packaged as an asynchronous FastAPI service and an interactive Streamlit UI with 16-bit DICOM ingestion, containerized with Docker, and backed by a comprehensive suite of 92 automated tests."
