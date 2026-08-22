# 🖼️ MedVision-AI: Portfolio Visual Assets & Screenshot Catalog

> **Portfolio screenshot gallery reference, recommended captions, and visual storytelling layout.**

---

## 📸 Key Portfolio Visuals

### 1. Interactive Diagnostic Dashboard (`app/streamlit_app.py`)
- **Visual:** Side-by-side display of original frontal radiograph and Grad-CAM activation heatmap overlay.
- **Caption:** *Streamlit diagnostic dashboard providing real-time pneumonia classification, operating threshold tuning ($t=0.60$), and Grad-CAM saliency localization over focal lung opacities.*

### 2. End-to-End System Architecture (`docs/architecture.md`)
- **Visual:** High-contrast Mermaid diagram illustrating the complete zero-leakage pipeline.
- **Caption:** *End-to-end engineering architecture from group-aware patient partitioning to two-stage DenseNet121 transfer learning and REST API serving.*

### 3. Receiver Operating Characteristic & Precision-Recall Curves
- **Visual:** Held-out test set ROC curve (0.8381 ROC-AUC) and PR curve (0.6022 PR-AUC).
- **Caption:** *Unbiased held-out test evaluation on 4,003 unique patients verifying robust discrimination and precision-recall trade-offs under 22.5% minority-class prevalence.*
