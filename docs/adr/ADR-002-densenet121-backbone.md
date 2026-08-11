# ADR-002: Choice of DenseNet121 as Primary Candidate Backbone

**Date:** 2026-08-11  
**Status:** Accepted (Candidate Selection)  

## Context
Medical chest X-ray images feature subtle opacity variations and localized structural features. Selecting an appropriate deep learning backbone requires evaluating feature reuse and parameter efficiency against baseline architectures.

## Decision
We select **DenseNet121** as our **primary candidate backbone model**, while keeping the model factory extensible to benchmark against **EfficientNetB0** and the **Custom CNN Baseline**.

## Rationale
- **Feature Concatenation & Reuse:** DenseNet connects each layer to every other layer in a feed-forward fashion, preserving high- and low-level feature maps.
- **Parameter Efficiency:** DenseNet121 has ~7M parameters (compared to ResNet50's ~25M), making it highly suitable for medical image feature extraction.
- **Extensible Factory:** Model selection remains configurable via `model.selected_architecture` in `config.yaml` (`densenet121`, `efficientnetb0`, `custom_cnn`).

## Consequences
- Image input shape defaults to `224 x 224 x 3` (single-channel grayscale X-rays expanded across 3 channels), but remains configurable.
