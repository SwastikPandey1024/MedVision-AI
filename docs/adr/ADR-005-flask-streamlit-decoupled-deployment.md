# ADR-005: Decoupled Flask REST API and Streamlit UI Architecture

**Date:** 2026-08-11  
**Status:** Accepted  

## Context
Deploying deep learning models requires separating backend model inference logic from user interface presentation to maintain modularity, testability, and scalablity.

## Decision
We decouple the system into two independent services:
1. **Flask REST API Service (`medvision.api`)**: Handles file validation, model loading, tensor preprocessing, inference, and Grad-CAM computation.
2. **Streamlit Application (`medvision.ui`)**: Pure frontend dashboard that communicates with the Flask backend via HTTP REST endpoints (`/predict`).

## Consequences
### Positive
- Allows the Flask API to be containerized and deployed independently to cloud platforms (AWS App Runner, GCP Cloud Run, Docker).
- Streamlit UI can be hosted on Streamlit Community Cloud without needing heavy TensorFlow dependencies installed in the frontend container.
- Clean separation of concerns simplifies unit testing.
