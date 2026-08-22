# 🔬 MedVision-AI: Phase 8 — Grad-CAM Explainability Engine

> **Documentation and technical audit of the Phase 8 Gradient-weighted Class Activation Mapping (Grad-CAM) explainability engine for DenseNet121 chest radiograph pneumonia detection.**

---

## 📌 1. Overview & Architectural Role

Deep neural networks deployed on medical imaging data require visual explainability to build trust with clinical and engineering stakeholders. Grad-CAM generates coarse 2D saliency maps highlighting the discriminative spatial regions in the input radiograph that influenced the model's classification score.

### Key Capabilities
- **Non-Mutating Gradient Model:** Extracts feature maps and computes target class gradients using `tf.GradientTape` without altering model weights.
- **Dynamic Layer Discovery:** Automatically identifies the optimal 4D convolutional feature layer (`conv5_block16_2_conv` in DenseNet121) or allows explicit user override.
- **Multimodal Visual Outputs:** Generates raw heatmaps, normalized heatmaps, blended color overlays ($\alpha = 0.40$), and side-by-side comparative panels.
- **Cross-Format Support:** Seamlessly ingests DICOM (`.dcm`), PNG, and JPEG inputs with proper VOI LUT windowing and aspect-preserving resizing.

---

## 🏗️ 2. Mathematical Foundation

For a target class $c$ (pneumonia) and convolutional feature map $A^k \in \mathbb{R}^{u \times v}$ from layer `conv5_block16_2_conv`:

1. **Class Gradients:**
   $$\frac{\partial y^c}{\partial A_{i, j}^k}$$

2. **Global Average Pooled Neuron Importance Weights:**
   $$\alpha_k^c = \frac{1}{u \cdot v} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i, j}^k}$$

3. **Rectified Linear Heatmap:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k \right)$$

4. **Normalization:**
   $$H_{\text{norm}} = \frac{L^c - \min(L^c)}{\max(L^c) - \min(L^c) + 10^{-8}}$$

---

## 💻 3. CLI Usage

Generate Grad-CAM visual assets from the command line:

```bash
# Single image execution
python scripts/gradcam.py \
  --checkpoint final_artifacts/densenet121_stage2_best.keras \
  --image test_dl.dcm \
  --threshold 0.60 \
  --output-dir artifacts/explainability

# Batch directory execution
python scripts/gradcam.py \
  --checkpoint final_artifacts/densenet121_stage2_best.keras \
  --batch-dir data/sample_dicoms \
  --threshold 0.60 \
  --output-dir artifacts/explainability
```

---

## 🧪 4. Unit & Regression Testing

The Grad-CAM engine is verified by `tests/test_gradcam.py`:
- `test_auto_detect_target_conv_layer`: Verifies dynamic discovery of last 4D feature map layer.
- `test_compute_gradcam_heatmap_valid_output`: Checks shape, range $[0.0, 1.0]$, and float32 dtype.
- `test_overlay_heatmap_rgb_and_dimensions`: Validates superimposition blending and dimensions.
- `test_overlay_heatmap_grayscale_input`: Confirms 2D grayscale image handling.
- `test_invalid_input_tensor_raises`: Verifies dimension validation and NaN/Inf rejection.
- `test_model_weights_not_mutated_during_gradcam`: Guarantees zero weight mutation.
- `test_generate_gradcam_explanation_pipeline`: Checks end-to-end dictionary contract.

Execute tests:
```bash
python -m pytest tests/test_gradcam.py -q
```
