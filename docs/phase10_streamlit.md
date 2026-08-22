# 🩺 MedVision-AI: Phase 10 — Streamlit Interactive Radiograph UI

> **Architecture, layout, and operation guide for the MedVision-AI Streamlit interactive web application.**

---

## 📌 1. Overview & UI Design Philosophy

The MedVision-AI Streamlit application provides an interactive, research-grade radiologist dashboard demonstrating model inference, threshold sensitivity, and Grad-CAM visual interpretability on chest radiographs.

### Design Principles
- **Modern Medical Dashboard Aesthetic:** High-contrast, clean typography, neutral grey and blue accents, zero cluttered elements.
- **Interactive Threshold Tuning:** Allows real-time visualization of how decision threshold adjustments impact positive/negative triage.
- **Side-by-Side Saliency Comparison:** Displays the input radiograph alongside the superimposed Grad-CAM activation overlay.
- **Persistent Non-Clinical Disclaimer:** Unmissable disclaimer ensuring transparent educational framing.

---

## 🏗️ 2. Application Architecture & Component Hierarchy

```
app/
├── streamlit_app.py           # Main application layout and event loop
├── components/
│   ├── header.py              # Hero title and technology badges
│   ├── disclaimer.py          # Regulatory disclaimer banner
│   └── metrics_card.py        # Verified held-out test benchmarks expander
└── services/
    └── inference_service.py   # Cached model loading & DICOM processing
```

### Visual Workflow
```
Radiograph Upload (DICOM / PNG / JPEG)
         │
         ▼
CR/DX Normalization & Bilinear Resizing (224, 224, 3)
         │
         ▼
DenseNet121 Inference ──► Pneumonia Probability Score
         │
         ▼
Validation-Frozen Thresholding (t = 0.60) ──► Flag / Normal Classification
         │
         ▼
Grad-CAM Gradient Tape (conv5_block16_2_conv) ──► Saliency Heatmap
         │
         ▼
Alpha Blending (α = 0.40) ──► Side-by-Side Saliency Visualizer
```

---

## 🚀 3. Running the Streamlit App

```bash
# Launch Streamlit interactive dashboard
streamlit run app/streamlit_app.py --server.port 8501
```

---

## 🧪 4. Interactive Features

1. **DICOM & Image Support:** Direct native support for 16-bit DICOM radiographs with VOI LUT windowing and standard PNG/JPEG uploads.
2. **Threshold Slider:** Adjust classification boundary from $0.10$ to $0.90$ (benchmarked optimal at $0.60$).
3. **Opacity Control:** Fine-tune heatmap transparency ($\alpha \in [0.0, 1.0]$).
4. **Built-in Demo Sample:** Instant 1-click loading of real sample DICOM radiograph.
