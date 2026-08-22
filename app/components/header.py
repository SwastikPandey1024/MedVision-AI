"""Header and hero UI component for Streamlit application."""

import streamlit as st


def render_header() -> None:
    """Render modern, clean portfolio header."""
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0 1rem 0;">
            <h1 style="font-size: 2.4rem; font-weight: 700; margin-bottom: 0.3rem;">
                🩺 MedVision-AI
            </h1>
            <p style="font-size: 1.15rem; color: #64748b; margin-bottom: 0.8rem;">
                Explainable Deep Learning System for Chest Radiograph Pneumonia Detection
            </p>
            <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                <span style="background: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">
                    DenseNet121 Transfer Learning
                </span>
                <span style="background: #dcfce7; color: #15803d; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">
                    ROC-AUC: 0.8381 (Held-Out Test)
                </span>
                <span style="background: #fef3c7; color: #b45309; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">
                    Frozen Threshold: 0.60
                </span>
                <span style="background: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">
                    Grad-CAM Visual Explainability
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
