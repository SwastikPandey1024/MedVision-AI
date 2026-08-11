# ADR-001: Selection of Keras 3 with TensorFlow Backend

**Date:** 2026-08-11  
**Status:** Accepted  

## Context
MedVision-AI requires a flexible, high-performance deep learning framework capable of managing image datasets, transfer learning backbones, layer fine-tuning, custom gradient computations (Grad-CAM), and deployment export.

## Decision
We select **Keras 3 running on top of TensorFlow 2.16+**.

## Consequences
### Positive
- Unified Keras 3 API syntax across high-level model construction and low-level gradient computation.
- Seamless compatibility with `tf.data` input processing pipelines for optimized memory prefetching.
- Native export formats (`.keras`) for simple model serialization and loading in Flask API servers.

### Negative
- Requires keeping dependencies synchronized with TensorFlow 2.16+ requirements.
