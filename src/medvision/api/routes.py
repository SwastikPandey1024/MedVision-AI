import time
from typing import Optional
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status
from fastapi.responses import JSONResponse

from medvision.api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
    ExplanationResponse,
    PredictAndExplainResponse,
    ErrorResponse,
    NON_CLINICAL_DISCLAIMER,
)
from medvision.api.services import (
    ModelService,
    decode_image_bytes,
    array_to_base64_png,
)
from medvision.explainability.gradcam import (
    generate_gradcam_explanation,
    compute_gradcam_heatmap,
    overlay_heatmap,
)
from medvision.utils.logger import get_logger

logger = get_logger("medvision.api.routes")
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["System"],
)
async def health_check():
    """Retrieve operational status, loaded model architecture, device, and frozen threshold."""
    service = ModelService.get_instance()
    return HealthResponse(
        status="healthy",
        service="MedVision-AI REST API",
        model_loaded=service.is_loaded(),
        model_name="DenseNet121",
        frozen_threshold=service.threshold,
        version="1.0.0",
        device=service.device,
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Model & Evaluation Metadata",
    tags=["System"],
)
async def get_metadata():
    """Retrieve detailed architectural parameters, validation benchmarks, and test metrics."""
    service = ModelService.get_instance()
    return MetadataResponse(
        model_architecture="DenseNet121",
        total_parameters=7301185,
        frozen_threshold=0.60,
        input_resolution="224x224x3",
        target_conv_layer=service.target_layer or "conv5_block16_2_conv",
        validation_metrics={
            "dataset": "RSNA Validation Split (4,003 unique patients)",
            "roc_auc": 0.8358,
            "pr_auc": 0.5944,
            "f1_score_at_0.60": 0.5898,
            "specificity_at_0.60": 0.8384,
            "sensitivity_at_0.60": 0.6408,
            "accuracy_at_0.60": 0.7939,
        },
        held_out_test_metrics={
            "dataset": "RSNA Held-Out Test Split (4,003 unique patients)",
            "roc_auc": 0.8381,
            "pr_auc": 0.6022,
            "f1_score_at_0.60": 0.5908,
            "accuracy_at_0.60": 0.7979,
            "sensitivity_at_0.60": 0.6475,
            "specificity_at_0.60": 0.8417,
            "precision_at_0.60": 0.5433,
            "patient_leakage": "0.0%",
        },
        non_clinical_disclaimer=NON_CLINICAL_DISCLAIMER,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Pneumonia Probability Inference",
    tags=["Inference"],
)
async def predict_pneumonia(
    file: UploadFile = File(..., description="Chest radiograph file (DICOM .dcm, PNG, JPEG)"),
    threshold: Optional[float] = Query(0.60, ge=0.0, le=1.0, description="Decision threshold override"),
):
    """Perform binary classification inference to estimate pneumonia probability.

    - Preprocesses radiograph to (224, 224, 3).
    - Applies frozen validation-optimized threshold ($t=0.60$).
    - Returns estimated probability, binary class flag, and elapsed runtime.
    """
    service = ModelService.get_instance()
    if not service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure checkpoint exists and service is initialized.",
        )

    t0 = time.perf_counter()
    contents = await file.read()

    try:
        _, preprocessed_tensor, fmt = decode_image_bytes(contents, file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    preds = service.model(preprocessed_tensor, training=False)
    prob = float(preds[0, 0].numpy())
    applied_threshold = threshold if threshold is not None else service.threshold
    is_pneumonia = bool(prob >= applied_threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        pneumonia_probability=round(prob, 4),
        predicted_class="Pneumonia" if is_pneumonia else "Normal",
        is_pneumonia=is_pneumonia,
        decision_threshold=applied_threshold,
        image_format=fmt,
        inference_time_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Grad-CAM Saliency Explanation",
    tags=["Explainability"],
)
async def explain_radiograph(
    file: UploadFile = File(..., description="Chest radiograph file"),
    alpha: float = Query(0.40, ge=0.0, le=1.0, description="Overlay blend opacity"),
    target_layer: Optional[str] = Query(None, description="Target layer override"),
):
    """Compute and return Grad-CAM saliency heatmaps and blended overlays."""
    service = ModelService.get_instance()
    if not service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Service is uninitialized.",
        )

    t0 = time.perf_counter()
    contents = await file.read()

    try:
        display_rgb, preprocessed_tensor, _ = decode_image_bytes(contents, file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    active_layer = target_layer or service.target_layer
    try:
        explanation = generate_gradcam_explanation(
            model=service.model,
            preprocessed_tensor=preprocessed_tensor,
            original_image=display_rgb,
            target_layer_name=active_layer,
            threshold=service.threshold,
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Grad-CAM error: {str(e)}")

    overlay_b64 = array_to_base64_png(explanation["overlay"])
    norm_hm_uint8 = np.uint8(np.clip(explanation["raw_heatmap"] * 255.0, 0, 255))
    heatmap_b64 = array_to_base64_png(norm_hm_uint8)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ExplanationResponse(
        pneumonia_probability=round(float(explanation["probability"]), 4),
        predicted_class=explanation["prediction"],
        decision_threshold=explanation["threshold"],
        target_layer=explanation["target_layer"],
        heatmap_base64=heatmap_b64,
        overlay_base64=overlay_b64,
        inference_time_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/predict-and-explain",
    response_model=PredictAndExplainResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Unified Prediction & Visual Explanation",
    tags=["Inference & Explainability"],
)
async def predict_and_explain(
    file: UploadFile = File(..., description="Chest radiograph file"),
    threshold: Optional[float] = Query(0.60, ge=0.0, le=1.0, description="Decision threshold"),
    alpha: float = Query(0.40, ge=0.0, le=1.0, description="Overlay blend opacity"),
):
    """Execute end-to-end inference and generate side-by-side explainability in a single call."""
    service = ModelService.get_instance()
    if not service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    t0 = time.perf_counter()
    contents = await file.read()

    try:
        display_rgb, preprocessed_tensor, _ = decode_image_bytes(contents, file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    applied_threshold = threshold if threshold is not None else service.threshold
    explanation = generate_gradcam_explanation(
        model=service.model,
        preprocessed_tensor=preprocessed_tensor,
        original_image=display_rgb,
        target_layer_name=service.target_layer,
        threshold=applied_threshold,
        alpha=alpha,
    )

    overlay_b64 = array_to_base64_png(explanation["overlay"])
    comp_b64 = array_to_base64_png(explanation["side_by_side"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return PredictAndExplainResponse(
        pneumonia_probability=round(float(explanation["probability"]), 4),
        predicted_class=explanation["prediction"],
        is_pneumonia=explanation["is_pneumonia"],
        decision_threshold=applied_threshold,
        target_layer=explanation["target_layer"],
        overlay_base64=overlay_b64,
        comparison_base64=comp_b64,
        inference_time_ms=round(elapsed_ms, 2),
    )
