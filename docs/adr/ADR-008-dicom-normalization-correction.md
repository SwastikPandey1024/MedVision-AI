# ADR-008: DICOM Intensity Normalization Correction for CR/DX Radiographs

**Date:** 2026-08-11  
**Status:** Accepted (Replaces CT HU Windowing in ADR-007)  

## Context
In initial iterations of Phase 2, DICOM image arrays were processed assuming Computed Tomography (CT) Hounsfield Unit (HU) calibration ($HU = \text{pixel} \times \text{RescaleSlope} + \text{RescaleIntercept}$) and soft-tissue windowing ($WC=40, WW=400$).

However, the RSNA Pneumonia Detection Challenge dataset consists of **plain projection chest radiographs (CR/DX modality)** rather than CT scans. CR/DX DICOM files do not have Hounsfield Unit calibration; `RescaleSlope` and `RescaleIntercept` are identity operations or non-calibrated values. Applying a CT soft-tissue window ($WC=40, WW=400$) to 12-bit raw radiograph pixel data ($0 - 4095$) artificially clipped over $90\%$ of real anatomical signal into solid white.

## Decision
We modify `src/medvision/data/dicom_utils.py` to implement **CR/DX Modality Intensity Normalization**:

1. **Tag-Based VOI LUT Linear Windowing**:
   - If DICOM `WindowCenter` ($WC$) and `WindowWidth` ($WW$) tags are present and valid ($WW > 0$), we apply the standard DICOM linear VOI LUT transformation directly to raw pixel values without HU conversion:
     $$y = \text{clip}\left(\frac{\text{pixel} - (WC - 0.5)}{WW - 1} + 0.5, 0, 1\right) \times 255.0$$

2. **Per-Image Percentile Clipping Fallback**:
   - If `WindowCenter` or `WindowWidth` tags are absent or invalid ($WW \le 0$), we fall back to robust per-image percentile clipping:
     $$y = \text{clip}\left(\frac{\text{pixel} - p_{0.5}}{\max(p_{99.5} - p_{0.5}, 1e-5)}, 0, 1\right) \times 255.0$$
     where $p_{0.5}$ and $p_{99.5}$ are the 0.5th and 99.5th percentiles of that specific radiograph's raw pixel array.

3. **Photometric Interpretation Handling**:
   - If DICOM tag `PhotometricInterpretation == "MONOCHROME1"`, the image array is inverted ($255 - y$) so bone structures appear white and lung air fields appear dark.

4. **Method Audit Tracking**:
   - Every parsed sample logs whether `tag_voi_lut` or `percentile_fallback` was used.

## Consequences
- Completely eliminates degenerate solid-black / solid-white image clipping.
- Preserves high-contrast pulmonary parenchymal details, rib cage structures, diaphragm contours, and focal opacities.
