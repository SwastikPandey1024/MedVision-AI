"""Pydantic data schemas for MedVision-AI REST API."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


NON_CLINICAL_DISCLAIMER = (
    "MedVision-AI is an academic research and educational demonstration tool. "
    "It is NOT a medical device and must NEVER be used for clinical diagnosis, "
    "screening, or patient management decisions."
)


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="API operational health status")
    service: str = Field("MedVision-AI REST API", description="Service identifier")
    model_loaded: bool = Field(..., description="Whether model weights are loaded in memory")
    model_name: str = Field("DenseNet121", description="Active model architecture")
    frozen_threshold: float = Field(0.60, description="Frozen decision threshold selected on validation data")
    version: str = Field("1.0.0", description="API service release version")
    device: str = Field("CPU/GPU", description="Inference compute device")


class MetadataResponse(BaseModel):
    model_architecture: str = "DenseNet121"
    total_parameters: int = 7301185
    frozen_threshold: float = 0.60
    input_resolution: str = "224x224x3"
    target_conv_layer: str = "conv5_block16_2_conv"
    validation_metrics: Dict[str, Any]
    held_out_test_metrics: Dict[str, Any]
    non_clinical_disclaimer: str = NON_CLINICAL_DISCLAIMER


class PredictionResponse(BaseModel):
    pneumonia_probability: float = Field(..., description="Estimated model pneumonia probability in [0, 1]")
    predicted_class: str = Field(..., description="'Pneumonia' if probability >= threshold, else 'Normal'")
    is_pneumonia: bool = Field(..., description="Boolean flag indicating positive threshold crossing")
    decision_threshold: float = Field(0.60, description="Decision threshold applied")
    image_format: str = Field(..., description="Detected image encoding format")
    inference_time_ms: float = Field(..., description="Elapsed inference time in milliseconds")
    model_version: str = Field("DenseNet121-Stage2-v1.0")
    non_clinical_disclaimer: str = NON_CLINICAL_DISCLAIMER


class ExplanationResponse(BaseModel):
    pneumonia_probability: float = Field(..., description="Estimated model probability")
    predicted_class: str = Field(..., description="Predicted class label")
    decision_threshold: float = Field(0.60, description="Decision threshold applied")
    target_layer: str = Field(..., description="Convolutional layer used for Grad-CAM gradients")
    heatmap_base64: Optional[str] = Field(None, description="Base64-encoded PNG of normalized saliency heatmap")
    overlay_base64: Optional[str] = Field(None, description="Base64-encoded PNG of blended radiograph overlay")
    inference_time_ms: float = Field(..., description="Total inference and explanation computation time in ms")
    non_clinical_disclaimer: str = NON_CLINICAL_DISCLAIMER


class PredictAndExplainResponse(BaseModel):
    pneumonia_probability: float = Field(..., description="Estimated model probability")
    predicted_class: str = Field(..., description="Predicted class label")
    is_pneumonia: bool = Field(..., description="Boolean flag indicating threshold crossing")
    decision_threshold: float = Field(0.60, description="Decision threshold applied")
    target_layer: str = Field(..., description="Convolutional layer used for Grad-CAM gradients")
    overlay_base64: str = Field(..., description="Base64-encoded PNG of blended radiograph overlay")
    comparison_base64: str = Field(..., description="Base64-encoded PNG of original vs overlay comparison")
    inference_time_ms: float = Field(..., description="Total processing time in ms")
    non_clinical_disclaimer: str = NON_CLINICAL_DISCLAIMER


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
