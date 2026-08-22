# 📸 MedVision-AI: Portfolio Screenshot Gallery & Capture Plan

> **Catalog of essential visual assets and screenshots for repository presentation, portfolio highlights, and LinkedIn case studies.**

---

## 🖼️ Portfolio Screenshot Specifications

| Screenshot ID | Target Filename | Dimensions | Description / What Must Be Visible | Recommended Documentation Placement |
| :--- | :--- | :--- | :--- | :--- |
| **SCREEN-01** | `hero_banner_medvision.png` | 1920 × 1080 | High-resolution hero graphic showcasing the Streamlit UI, DenseNet121 architecture badge, and test performance. | `README.md` Top Hero, Portfolio Landing |
| **SCREEN-02** | `system_architecture_full.png` | 1600 × 900 | Complete Mermaid pipeline: RSNA Ingestion $\to$ 0% Leakage Splitting $\to$ Two-Stage DenseNet121 $\to$ Forensics $\to$ Evaluation. | `README.md`, `docs/architecture.md` |
| **SCREEN-03** | `training_val_curves_stage1_stage2.png` | 1400 × 700 | Comparative loss and PR-AUC convergence trajectories for Stage 1 (feature extraction) and Stage 2 (fine-tuning). | `docs/portfolio.md`, `docs/final_metrics.md` |
| **SCREEN-04** | `final_held_out_metrics_table.png` | 1200 × 800 | Full verified test set metrics table showing 0.8381 ROC-AUC, 0.6022 PR-AUC, 84.17% Specificity on 4,003 patients. | `README.md`, `docs/final_metrics.md` |
| **SCREEN-05** | `held_out_roc_curve.png` | 1000 × 1000 | Receiver Operating Characteristic (ROC) curve with annotated ROC-AUC = 0.8381. | `docs/final_metrics.md`, `README.md` |
| **SCREEN-06** | `held_out_pr_curve.png` | 1000 × 1000 | Precision-Recall curve with annotated PR-AUC = 0.6022 under 22.5% prevalence. | `docs/final_metrics.md`, `README.md` |
| **SCREEN-07** | `held_out_confusion_matrix.png` | 1000 × 1000 | Confusion matrix @ $t=0.60$ (TP=584, TN=2610, FP=491, FN=318). | `docs/portfolio.md`, `README.md` |
| **SCREEN-08** | `threshold_selection_scan_audit.png` | 1200 × 600 | 81-point threshold scan plot highlighting validation F1 optimization peak at $t=0.60$. | `docs/final_metrics.md` |
| **SCREEN-09** | `streamlit_upload_prediction_ui.png` | 1920 × 1080 | Streamlit dashboard showing radiograph upload, probability gauge (27.19%), and normal classification badge. | `README.md`, `docs/phase10_streamlit.md` |
| **SCREEN-10** | `streamlit_gradcam_overlay_view.png` | 1920 × 1080 | Streamlit UI displaying side-by-side original radiograph vs. Grad-CAM saliency overlay over lung opacity. | `README.md`, `docs/portfolio.md` |
| **SCREEN-11** | `gradcam_quadrant_case_studies.png` | 1600 × 1200 | 4-panel diagnostic comparison: True Positive, True Negative, False Positive, and False Negative Grad-CAMs. | `docs/portfolio.md`, `docs/phase8_gradcam.md` |
| **SCREEN-12** | `fastapi_swagger_interactive_docs.png` | 1400 × 900 | Interactive OpenAPI `/docs` page displaying `/health`, `/metadata`, `/predict`, `/explain`, and `/predict-and-explain`. | `docs/phase9_api.md` |

---

## 🎨 Asset Capture & Quality Guidelines

1. **Window Sizing & Scaling:** Capture all desktop dashboard screenshots at minimum 1080p ($1920 \times 1080$) at 100% or 125% DPI scale.
2. **Theme Consistency:** Use GitHub-compatible high-contrast dark or light backgrounds.
3. **Clinical Authenticity:** Always retain aspect ratios on radiograph images to avoid distortion of lung fields.
