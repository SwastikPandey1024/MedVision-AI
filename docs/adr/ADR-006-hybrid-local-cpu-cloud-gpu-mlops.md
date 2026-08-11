# ADR-006: Hybrid Local CPU Development & Cloud GPU Training MLOps Strategy

**Date:** 2026-08-11  
**Status:** Accepted  

## Context
The local development environment is a Windows laptop powered by an Intel Core i5-13500H CPU with 16 GB RAM and integrated graphics (no NVIDIA CUDA GPU). Full deep-learning model training on the complete RSNA dataset (~26,000+ high-resolution chest X-rays) requires dedicated GPU acceleration.

## Decision
We adopt a **Hybrid MLOps Workflow Strategy**:
1. **Local Laptop (Control & Development Environment)**:
   - Python 3.11 with standard CPU-based TensorFlow.
   - Code authoring, pipeline architecture, API/UI development, and unit testing.
   - Fast CPU smoke tests executed in `execution_mode: "development"` using a small dataset sample (`sample_fraction_dev: 0.05`).
2. **Cloud GPU (Training Environment - Kaggle / Google Colab)**:
   - Full dataset model training, transfer learning, fine-tuning, and experiment sweeps executed in `execution_mode: "full"`.
3. **GitHub Repository (Single Source of Truth)**:
   - All code, pipeline definitions, YAML configs, unit tests, and ADRs are versioned centrally on GitHub.
   - Model weights (`.keras`) are saved to cloud artifact storage and **never** committed to version control.

## Consequences
### Positive
- Prevents local laptop hardware thermal throttling or out-of-memory crashes.
- Guarantees 100% reproducible training execution across local and cloud environments.
- Keeps repository lightweight and free of dataset/model binary pollution.

### Negative
- Cloud training runs require pushing code to GitHub/pulling in Colab/Kaggle or setting up repository sync.
