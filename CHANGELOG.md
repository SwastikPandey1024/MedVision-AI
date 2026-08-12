# Changelog

All notable changes to the **MedVision-AI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0-alpha] - 2026-08-12

### Fixed & Added
- **Single-Instance Distribution Strategy Architecture & Kaggle Multi-GPU Fix**:
  - Refactored `build_model` in `src/medvision/models/factory.py` to accept `strategy: Optional[tf.distribute.Strategy] = None` and reuse existing `MirroredStrategy` instances without secondary creation.
  - Updated `scripts/train.py` to instantiate `strategy` once and pass it downstream to eliminate `RuntimeError: Mixing different tf.distribute.Strategy objects`.
  - Added explicit Strategy object reuse assertion logging: `Strategy object identity / reuse: PASS`.
  - Enforced persistent directory locking `%cd /kaggle/working/MedVision-AI` across Kaggle notebook cells in `notebooks/kaggle/medvision_ai_kaggle_gpu.ipynb`.
- **Validation-Only Decision Threshold Selection & Evaluation Engine**:
  - Implemented `select_optimal_threshold_from_val` in `src/medvision/evaluation/threshold.py` to lock thresholds strictly on validation data.
  - Implemented `generate_model_comparison_report` in `src/medvision/evaluation/reporting.py` with primary ranking on **PR-AUC** and secondary on **ROC-AUC**.
  - Implemented experiment manifest reproducibility generator in `src/medvision/utils/reproducibility.py`.
  - Added comprehensive test suites in `tests/test_evaluation.py` and `tests/test_models.py` (30 passed tests).
  - Documented multi-GPU distribution architecture in `docs/adr/ADR-009-multi-gpu-distribution-strategy-reuse.md`.

## [0.4.0-alpha] - 2026-08-11

### Added
- **Phase 3 Baseline Custom CNN Architecture & Kaggle GPU Training Pipeline**:
  - Implemented lightweight `CustomCNNBaseline` model (`build_custom_cnn`) with 3 Conv-BN-ReLU-Pool-Dropout blocks in `src/medvision/models/baseline_cnn.py`.
  - Implemented `DenseNet121Primary` transfer learning builder and `unfreeze_densenet_for_finetuning` with explicit `BatchNormalization` layer freeze safety in `src/medvision/models/densenet.py`.
  - Built multi-GPU hardware distribution & mixed precision model factory in `src/medvision/models/factory.py`.
  - Implemented `train_model` engine with class weights, `ModelCheckpoint` monitoring `val_pr_auc`, `EarlyStopping`, `ReduceLROnPlateau`, and CSV/TensorBoard loggers in `src/medvision/models/trainer.py`.
  - Created custom metric suite (`Specificity`, `F1Score`, `PR-AUC`, `ROC-AUC`, `Precision`, `Recall`) in `src/medvision/utils/metrics.py`.
  - Added training command-line interface script `scripts/train.py`.
  - Documented architectural decisions in `docs/adr/ADR-009-baseline-cnn-architecture.md`.
  - Updated unit test suite in `tests/test_models.py` (21 passed tests).

## [0.3.0-alpha] - 2026-08-11

### Added
- **Phase 2 Preprocessing Engine & `tf.data` Pipeline**:
  - Implemented DICOM RescaleSlope/Intercept and Lung Windowing ($WC=40, WW=400$) in `src/medvision/data/dicom_utils.py`.
  - Implemented one-time TFRecord serializer and sharding engine in `src/medvision/data/tfrecord_writer.py`.
  - Implemented high-performance `tf.data.Dataset` input pipeline with augmentation ops in `src/medvision/data/preprocessing.py`.
  - Added strict Anatomical Augmentation Policy document in `docs/adr/ADR-007-augmentation-policy.md`.
  - Created lightweight PIL/OpenCV dev loader `src/medvision/data/local_dev_loader.py` for 5% CPU dev-subset testing.
  - Added unit test suites in `tests/test_dicom_utils.py`, `tests/test_tfrecords.py`, and `tests/test_preprocessing.py` (15 passed tests).

## [0.2.0-alpha] - 2026-08-11

### Added
- **Phase 1 Dataset Acquisition & Data Engineering**:
  - Implemented dynamic dataset root auto-detection in `src/medvision/data/dataset.py`.
  - Implemented RSNA metadata parser `parse_rsna_manifest` aggregating bounding boxes per patient.
  - Implemented data quality and integrity auditor `src/medvision/data/validation.py`.
  - Implemented patient-aware group splitter `create_patient_aware_splits` and zero-leakage verifier `verify_zero_patient_leakage` in `src/medvision/data/splits.py`.
  - Implemented EDA statistics generator and Markdown/JSON report exporter `src/medvision/data/eda.py`.
  - Added synthetic unit tests in `tests/test_data.py`.
  - Updated Kaggle GPU notebook with Phase 1 execution cells.

## [0.1.0-alpha] - 2026-08-11

### Added
- **Phase 0 Foundational Infrastructure**:
  - Established modular Python package architecture in `src/medvision`.
  - Configured `pyproject.toml`, `requirements.txt`, and `requirements-dev.txt`.
  - Created Product Requirements Document (`docs/PRD.md`), Business Requirements Document (`docs/BRD.md`), and Software Requirements Specification (`docs/SRS.md`).
  - Added Architecture Decision Records (`docs/adr/ADR-001` through `ADR-006`).
  - Added repository governance files (`README.md`, `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `.gitignore`, `.env.example`).
