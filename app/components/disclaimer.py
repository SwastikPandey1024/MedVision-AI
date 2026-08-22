"""Non-clinical disclaimer banner for Streamlit UI."""

import streamlit as st


def render_disclaimer() -> None:
    """Render persistent medical disclaimer banner."""
    st.info(
        "⚠️ **Research & Educational Demonstration Only**: MedVision-AI is an academic deep learning "
        "engineering prototype. It is **not** an FDA/CE-cleared medical device and must **never** be used "
        "for clinical diagnosis, medical triage, or patient management decisions."
    )
