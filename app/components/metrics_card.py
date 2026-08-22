"""Metrics summary card component for Streamlit application."""

import streamlit as st


def render_performance_summary() -> None:
    """Render verified test benchmarks table."""
    with st.expander("📊 Model Architecture & Verified Held-Out Test Benchmarks (4,003 Patients)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", "0.8381", "Held-Out Test")
        c2.metric("PR-AUC", "0.6022", "22.5% Prevalence")
        c3.metric("Specificity", "84.17%", "+5.20% vs t=0.50")
        c4.metric("Accuracy", "79.79%", "@ Threshold 0.60")

        st.markdown(
            """
            | Benchmark Property | Specification / Value | Engineering Context |
            | :--- | :--- | :--- |
            | **Model Architecture** | DenseNet121 | 7,301,185 parameters (Two-Stage Transfer Learning) |
            | **Patient Leakage** | **0.0%** | Stratified Group Splitting on `patient_id` |
            | **Frozen Decision Threshold** | **$t = 0.60$** | Selected strictly on validation data to optimize F1 |
            | **Held-Out Test Cohort** | 4,003 unique patients | RSNA Pneumonia Detection Challenge |
            | **Saliency Mapping** | Grad-CAM | Feature layer: `conv5_block16_2_conv` |
            """
        )
