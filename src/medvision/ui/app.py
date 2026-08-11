"""Streamlit Web Application dashboard entrypoint (Phase 10)."""

import streamlit as st


def main() -> None:
    """Render Streamlit frontend interface."""
    st.set_page_config(
        page_title="MedVision-AI: Pneumonia Detection",
        page_icon="🩺",
        layout="wide",
    )

    st.title("🩺 MedVision-AI: Explainable Pneumonia Detection System")
    st.caption("AI-Assisted Research & Educational Tool | Not for Clinical Use")

    st.warning(
        "⚠️ **Disclaimer**: MedVision-AI is an educational research prototype. "
        "It has not been cleared by FDA/CE for clinical diagnostic use."
    )

    st.info("Phase 0 Initialization complete. Streamlit interactive UI will be implemented in Phase 10.")


if __name__ == "__main__":
    main()
