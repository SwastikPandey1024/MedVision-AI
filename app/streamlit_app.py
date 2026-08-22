"""MedVision-AI Interactive Streamlit Web Application Dashboard (Phase 10)."""

import sys
from pathlib import Path

# Ensure ROOT_DIR and src/ are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = str(ROOT_DIR / "src")
APP_DIR = str(ROOT_DIR / "app")
for p in [str(ROOT_DIR), SRC_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import numpy as np
from PIL import Image

try:
    from app.components.header import render_header
    from app.components.disclaimer import render_disclaimer
    from app.components.metrics_card import render_performance_summary
    from app.services.inference_service import get_cached_model, process_uploaded_image
except ImportError:
    from components.header import render_header
    from components.disclaimer import render_disclaimer
    from components.metrics_card import render_performance_summary
    from services.inference_service import get_cached_model, process_uploaded_image

from medvision.explainability.gradcam import generate_gradcam_explanation


def main():
    st.set_page_config(
        page_title="MedVision-AI | Chest Radiograph Analysis",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Render Header & Non-Clinical Disclaimer
    render_header()
    render_disclaimer()

    # 2. Sidebar Controls
    st.sidebar.header("⚙️ Inference Configuration")
    threshold = st.sidebar.slider(
        "Decision Threshold ($t$)",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.01,
        help="Frozen validation-optimal operating threshold is 0.60.",
    )
    if threshold != 0.60:
        st.sidebar.caption("⚠️ Note: Official benchmarked threshold is **0.60**.")

    alpha = st.sidebar.slider(
        "Grad-CAM Heatmap Blend ($\alpha$)",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Opacity of the color heatmap blended over the radiograph.",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("📁 Radiograph Source")
    input_mode = st.sidebar.radio(
        "Select Input Source",
        ["Upload File (DICOM / PNG / JPEG)", "Use Demo Sample Radiograph"],
    )

    # 3. Model Loading
    try:
        model, target_layer = get_cached_model()
        st.sidebar.success(f"✅ Model Loaded: DenseNet121\nTarget Layer: `{target_layer}`")
    except Exception as e:
        st.sidebar.error(f"❌ Model load error: {e}")
        st.error(f"Could not initialize DenseNet121 model: {e}")
        st.stop()

    file_bytes = None
    file_name = ""

    if input_mode == "Use Demo Sample Radiograph":
        sample_path = ROOT_DIR / "test_dl.dcm"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                file_bytes = f.read()
            file_name = sample_path.name
            st.sidebar.info(f"Loaded demo DICOM: `{file_name}`")
        else:
            st.sidebar.warning("Demo sample not found on disk. Please upload an image.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Frontal Chest Radiograph",
            type=["dcm", "dicom", "png", "jpg", "jpeg", "bmp"],
            help="Upload an adult frontal chest radiograph.",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

    st.markdown("---")

    # 4. Main Analysis Workflow
    if file_bytes is not None:
        with st.spinner("Processing radiograph and computing Grad-CAM saliency..."):
            try:
                display_img, preprocessed_tensor = process_uploaded_image(file_bytes, file_name)

                explanation = generate_gradcam_explanation(
                    model=model,
                    preprocessed_tensor=preprocessed_tensor,
                    original_image=display_img,
                    target_layer_name=target_layer,
                    threshold=threshold,
                    alpha=alpha,
                )
            except Exception as err:
                st.error(f"Error during image processing or inference: {err}")
                st.stop()

        prob = float(explanation["probability"])
        is_pneumonia = bool(explanation["is_pneumonia"])
        pred_label = explanation["prediction"]

        # Results Summary Section
        st.subheader("🎯 Model Prediction & Operating Point Analysis")

        col_metric1, col_metric2, col_metric3 = st.columns([1.5, 2, 1.5])

        with col_metric1:
            if is_pneumonia:
                st.error(f"### 🚩 Flagged: {pred_label}")
                st.caption(f"Estimated Probability >= {threshold:.2f}")
            else:
                st.success(f"### 🛡️ Classified: {pred_label}")
                st.caption(f"Estimated Probability < {threshold:.2f}")

        with col_metric2:
            st.markdown(f"**Estimated Pneumonia Probability:** `{prob:.2%}` ({prob:.4f})")
            st.progress(prob)
            st.caption(f"Operating Decision Threshold: **{threshold:.2f}** (Validation-Frozen: **0.60**)")

        with col_metric3:
            st.metric(
                label="Decision Margin",
                value=f"{abs(prob - threshold):.2%}",
                delta=f"{'+' if prob >= threshold else '-'}{abs(prob - threshold):.2%} vs threshold",
                delta_color="normal" if is_pneumonia else "inverse",
            )

        st.markdown("---")

        # Visual Comparison Section
        st.subheader("🔍 Visual Explainability: Original Radiograph vs. Grad-CAM Saliency Overlay")

        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown("#### Original Frontal Radiograph")
            st.image(
                explanation["original_image"],
                caption=f"Input: {file_name} ({display_img.shape[1]} × {display_img.shape[0]} px)",
                use_container_width=True,
            )

        with col_img2:
            st.markdown(f"#### Grad-CAM Saliency Overlay (Layer: `{target_layer}`)")
            st.image(
                explanation["overlay"],
                caption=f"Superimposed Heatmap (Blend $\\alpha = {alpha:.2f}$) | Prob: {prob:.1%}",
                use_container_width=True,
            )

        with st.expander("🔬 View Raw Grad-CAM Activation Heatmap", expanded=False):
            raw_hm = (explanation["raw_heatmap"] * 255.0).astype(np.uint8)
            st.image(
                raw_hm,
                caption=f"Normalized Activation Grid (Resolution: {raw_hm.shape[1]} × {raw_hm.shape[0]})",
                width=300,
            )

    else:
        st.info("👈 Please upload a chest radiograph or select the demo sample in the sidebar to begin analysis.")

    st.markdown("---")
    # 5. Architecture & Performance Benchmarks
    render_performance_summary()


if __name__ == "__main__":
    main()
