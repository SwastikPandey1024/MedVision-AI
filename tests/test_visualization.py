"""Unit tests for model architecture visualization utilities."""

import pytest
import os
from pathlib import Path
import keras
import tensorflow as tf

from medvision.utils.visualization import visualize_architecture, generate_model_summary_txt, generate_model_architecture_diagram
from medvision.models.factory import build_model


@pytest.mark.parametrize("architecture", ["custom_cnn", "densenet121", "efficientnetb0"])
def test_visualize_architecture_supported_models(tmp_path, architecture):
    """Verify visualization generation for custom_cnn, densenet121, and efficientnetb0 without GPU."""
    output_dir = tmp_path / "arch_test"
    
    result = visualize_architecture(architecture=architecture, output_dir=output_dir)

    assert result["architecture"] == architecture
    assert result["output_shape"] == (None, 1)
    assert result["total_params"] > 0
    assert result["trainable_params"] >= 0

    # Verify generated artifact files exist and are non-empty
    txt_file = result["summary_txt_path"]
    svg_file = result["svg_path"]
    png_file = result["png_path"]

    assert txt_file.exists() and txt_file.stat().st_size > 0
    assert svg_file.exists() and svg_file.stat().st_size > 0
    assert png_file.exists() and png_file.stat().st_size > 0

    # Check contents of text summary
    summary_content = txt_file.read_text(encoding="utf-8")
    assert "MEDVISION-AI MODEL ARCHITECTURE REPORT" in summary_content
    assert "PARAMETER COUNT SUMMARY" in summary_content
    assert f"Total Parameters      : {result['total_params']:,}" in summary_content


def test_generate_model_summary_txt_structure(tmp_path):
    """Verify model summary TXT file structure and layer details."""
    model = build_model("custom_cnn", compile_model=False)
    txt_path = tmp_path / "custom_cnn_test_summary.txt"

    content = generate_model_summary_txt(model, txt_path)

    assert txt_path.exists()
    assert "LAYER SUMMARY TABLE" in content
    assert "CustomCNNBaseline" in content
    assert "Total Parameters" in content


def test_visualization_gpu_independent(tmp_path):
    """Verify visualization generation runs seamlessly on CPU without GPU hardware."""
    # Force TensorFlow to hide physical GPU devices if present
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    output_dir = tmp_path / "cpu_arch_test"
    result = visualize_architecture(architecture="custom_cnn", output_dir=output_dir)

    assert result["svg_path"].exists()
    assert result["png_path"].exists()
