# ADR-003: Enforcement of Patient-Aware Group Splitting

**Date:** 2026-08-11  
**Status:** Accepted  

## Context
In medical datasets like RSNA Pneumonia Challenge, multiple X-ray images often belong to the same patient (taken over different visits or follow-ups). Naive random splitting causes data leakage where images from the same patient appear in both training and test sets.

## Decision
We mandate **Patient-Aware Group Splitting** (`GroupKFold` or `GroupShuffleSplit` on `PatientID`) for all train/validation/test partitions.

## Consequences
### Positive
- Completely eliminates patient data leakage between training and evaluation splits.
- Guarantees realistic out-of-sample test set metrics that accurately reflect performance on new, unseen patients.

### Negative
- Requires maintaining a strict patient ID manifest and metadata mapping during ingestion.
