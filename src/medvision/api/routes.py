"""Flask API route definitions for /health and /predict (Phase 9)."""

from flask import Blueprint, jsonify, request

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify API and model status."""
    return jsonify({
        "status": "healthy",
        "service": "MedVision-AI API",
        "version": "0.1.0-alpha",
        "model_loaded": False,
    }), 200


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Pneumonia prediction endpoint receiving image payload.

    Returns:
        JSON response with prediction label, probability score, and Grad-CAM base64 image string.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image payload provided."}), 400

    return jsonify({
        "status": "stub",
        "message": "Prediction endpoint will be implemented in Phase 9.",
    }), 501
