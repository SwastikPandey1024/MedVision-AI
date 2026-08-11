# ADR-007: Image Data Augmentation Policy & Anatomical Constraints

**Date:** 2026-08-11  
**Status:** Accepted  

## Context
In deep learning for medical radiograph analysis, data augmentation reduces overfitting. However, applying arbitrary geometric transformations can alter anatomical relationships or invalidate diagnostic features.

## Decision
We establish a **Strict Anatomical Augmentation Policy**:

1. **Permitted Transformations**:
   - **Horizontal Flipping (`random_flip_left_right`)**: Clinically valid for binary Pneumonia detection (Normal vs Pneumonia).
   - **Slight Rotation ($\pm 10^\circ$)**: Accounts for minor patient positioning variance during X-ray acquisition.
   - **Zoom / Scaling ($\pm 10\%$)**: Accounts for variations in patient-to-detector distance.
   - **Brightness & Contrast Jitter ($\pm 8\%$)**: Simulates variations in X-ray tube current and exposure timing.

2. **Forbidden Transformations**:
   - **Vertical Flipping (`random_flip_up_down`)**: **Strictly Forbidden.** Inverts the apex-to-base pulmonary orientation and diaphragm positioning.
   - **Extreme Shear / Distortion**: **Strictly Forbidden.** Distorts anatomical rib and lung field geometry.

## Scope Limitation & Warning
> [!CAUTION]
> **Laterality Scope Limitation**: Horizontal flipping is acceptable **only** for symmetrical binary findings (Normal vs Pneumonia). If MedVision-AI is ever expanded to detect laterality-dependent pathology (e.g. dextrocardia, situs inversus, or left vs. right lung lesion tracking), **horizontal flipping must be immediately removed** from the augmentation pipeline.

## Consequences
- Preserves natural chest radiograph geometry while enhancing model generalization.
