# ADR-004: Integration of Grad-CAM for Spatial Explainability

**Date:** 2026-08-11  
**Status:** Accepted  

## Context
Black-box neural network predictions are unacceptable in healthcare research. Users need to understand *which spatial regions* of a chest X-ray contributed to a pneumonia prediction.

## Decision
We implement **Gradient-weighted Class Activation Mapping (Grad-CAM)** targeted at the final dense block of DenseNet121 (`conv5_block16_concat`).

## Rationale
- Grad-CAM utilizes gradients flowing into the final convolutional layer to produce a coarse localization map highlighting important image regions.
- Requires no architectural modification or re-training of the classifier.

## Consequences
- Requires custom GradientTape extraction during inference in TensorFlow/Keras.
- Heatmaps must be clearly documented as visual AI model focus indicators, **not** certified clinical lesion boundary demarcations.
