# 💼 MedVision-AI: LinkedIn Project Descriptions & Posts

Use these descriptions for LinkedIn project entries, portfolio links, or technical showcase posts.

---

## 📱 Short Version (Quick Post / Profile Project Summary)

Excited to share **MedVision-AI**, an end-to-end deep learning system for pneumonia detection from chest radiographs using the RSNA dataset (26,684 patient studies).

Rather than just training a baseline classifier, I focused on rigorous ML engineering:
🔹 **Zero-Leakage Data Partitioning:** Group-stratified splitting on `patient_id` ensuring 0.0% patient overlap across train, validation, and test sets (4,003 test patients).
🔹 **Two-Stage Transfer Learning:** DenseNet121 feature extraction + fine-tuning with strictly frozen Batch Normalization layers to protect pre-trained statistics.
🔹 **Validation-Isolated Threshold Optimization:** Calibrated the decision threshold ($t=0.60$) exclusively on validation data, achieving **84.17% Specificity**, **79.79% Accuracy**, and **0.8381 ROC-AUC** on the held-out test split.

Check out the full open-source codebase, architecture diagrams, and reproducibility guide on GitHub:
👉 https://github.com/SwastikPandey1024/MedVision-AI

#MachineLearning #DeepLearning #ComputerVision #MedicalAI #TensorFlow #Python #MLEngineering

---

## 📝 Medium Version (Technical Deep-Dive Post)

In medical computer vision, modeling decisions are only as good as your evaluation discipline. Patient leakage, fine-tuning instability, and data snooping often lead to falsely optimistic benchmark claims that fail in the real world.

To address these challenges, I built **MedVision-AI** — a production-ready pneumonia classification pipeline trained on 26,684 chest X-rays from the RSNA Pneumonia Detection Challenge.

Here are the key engineering highlights:

1️⃣ **Patient-Aware Stratification (0% Leakage):** Splitting medical images at the image level leaks patient anatomy across splits. MedVision-AI implements target-stratified group k-fold partitioning on `patient_id`, creating an uncontaminated 4,003-patient held-out test set.

2️⃣ **Two-Stage DenseNet121 Architecture:** Feature extraction followed by controlled unfreezing of the top 20 layers. Crucially, Batch Normalization layers were frozen during fine-tuning to prevent small-batch noise from destroying ImageNet channel calibrations.

3️⃣ **Numerical Stability Forensics:** Profiling loss dynamics and gradients across FP32 and `mixed_float16` to guarantee absence of gradient spikes or precision-related anomalies.

4️⃣ **Zero-Snooping Threshold Selection:** Decision threshold search ($t=0.60$) was performed exclusively on validation data. Applied to the held-out test set, it delivered:
- **ROC-AUC:** `0.8381`
- **PR-AUC:** `0.6022`
- **Specificity:** `84.17%` (reducing false positive rate by 24.7%)
- **Accuracy:** `79.79%`

The complete pipeline is modular, fully unit-tested, and reproducible:
🔗 https://github.com/SwastikPandey1024/MedVision-AI

Feedback and discussions welcome!

#AI #MachineLearning #ComputerVision #HealthcareAI #DataScience #Python #TensorFlow #MLOps

---

## 📖 Long Version (Comprehensive Article / Project Showcase)

### Building MedVision-AI: Engineering a Leakage-Free Medical Computer Vision Pipeline

When developing machine learning models for healthcare imaging, the most dangerous pitfalls aren't model depth or hyperparameter tuning — they are silent data leakage and evaluation bias.

I built **MedVision-AI**, an open-source deep learning system designed for pneumonia detection from frontal chest X-rays using 26,684 radiographs from the RSNA Pneumonia Detection Challenge.

Here is a breakdown of the technical decisions and engineering system behind it:

#### 1. Data Engineering & Group-Aware Splitting
In hospital workflows, patients often receive multiple sequential radiographs. Randomly partitioning images causes images of the same patient to appear in both training and test sets — leading the model to memorize patient-specific ribcage contours rather than pathological consolidation. MedVision-AI implements target-stratified group partitioning on `patient_id`:
- Train Set: 18,678 patients (70%)
- Validation Set: 4,003 patients (15%)
- Test Set: 4,003 patients (15%)
- Result: **0.0% patient leakage mathematically verified by automated tests.**

#### 2. Architecture & Staged Training Strategy
We leveraged DenseNet121 pre-trained on ImageNet. Because medical features require both fine-grained textural detail and global spatial context, DenseNet’s feature reuse across dense blocks provides ideal inductive bias.
- **Stage 1 (Feature Extraction):** Locked backbone weights, training a customized classification head with Adam ($LR=10^{-4}$, clipnorm=1.0).
- **Stage 2 (Fine-Tuning):** Selectively unfroze the top 20 convolutional layers ($LR=10^{-5}$). **Critical choice:** All Batch Normalization layers remained locked (`trainable=False`) to avoid small-batch gradient noise destabilizing pre-trained moving statistics.

#### 3. Numerical Forensics & Precision Profiling
We ran comparative profiling between full-precision (FP32) and 16-bit mixed-precision (`mixed_float16`) on multi-GPU instances. We verified exact convergence equivalence, zero exploding gradient events, and zero NaN occurrences.

#### 4. Threshold Optimization with Zero Test Snooping
Rather than reporting default 0.50 thresholds or tuning on test data, we conducted an 81-point grid search over validation predictions alone. Selecting $t=0.60$ optimized the operating point for screening workflows:
- **ROC-AUC (Primary):** `0.8381`
- **PR-AUC (Primary):** `0.6022`
- **Specificity:** `84.17%` (161 fewer false alarms than $t=0.50$)
- **Accuracy:** `79.79%`
- **Sensitivity:** `64.75%`

#### 5. Clean Code, Testing & Reproducibility
- Complete test coverage across data loading, group splitting, architecture, and evaluation.
- Deterministic 5% synthetic/development data loader for rapid local CPU iteration.
- Structured documentation, architecture diagrams, and execution guides.

Explore the code, architecture, and documentation here:
👉 https://github.com/SwastikPandey1024/MedVision-AI

*(Note: MedVision-AI is an academic/educational research project and not a clinical medical device.)*
