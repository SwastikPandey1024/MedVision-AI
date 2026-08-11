# Changelog

All notable changes to the **MedVision-AI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-alpha] - 2026-08-11

### Added
- **Phase 1 Dataset Acquisition & Data Engineering**:
  - Implemented dynamic dataset root auto-detection in `src/medvision/data/dataset.py`.
  - Implemented RSNA metadata parser `parse_rsna_manifest` aggregating bounding boxes per patient.
  - Implemented data quality and integrity auditor `src/medvision/data/validation.py`.
  - Implemented patient-aware group splitter `create_patient_aware_splits` and zero-leakage verifier `verify_zero_patient_leakage` in `src/medvision/data/splits.py`.
  - Implemented EDA statistics generator and Markdown/JSON report exporter `src/medvision/data/eda.py`.
  - Added synthetic unit tests in `tests/test_data.py` (Passed 7 unit tests).
  - Updated Kaggle GPU notebook with Phase 1 execution cells.

## [0.1.0-alpha] - 2026-08-11

### Added
- **Phase 0 Foundational Infrastructure**:
  - Established modular Python package architecture in `src/medvision`.
  - Configured `pyproject.toml`, `requirements.txt`, and `requirements-dev.txt`.
  - Created Product Requirements Document (`docs/PRD.md`), Business Requirements Document (`docs/BRD.md`), and Software Requirements Specification (`docs/SRS.md`).
  - Added Architecture Decision Records (`docs/adr/ADR-001` through `ADR-006`).
  - Added repository governance files (`README.md`, `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `.gitignore`, `.env.example`).
