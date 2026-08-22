# 🔬 MedVision-AI: Engineering & Research Notes

> **Technical journal documenting algorithmic decisions, empirical experiments, and design rationale across the 13-phase engineering lifecycle.**

---

## 📑 1. Architectural Justifications

### DenseNet121 vs. Standard ResNet Backbones
In radiograph interpretation, pathological consolidation often presents as subtle textural haziness across varying spatial scales. DenseNet's dense connectivity pattern concatenates all preceding feature maps within each dense block:
- **Feature Preservation:** Low-level edge and texture representations are directly accessible to deep classifier layers.
- **Gradient Flow:** Direct supervision signal is propagated back through identity shortcuts, preventing vanishing gradients during fine-tuning.
- **Parameter Efficiency:** DenseNet121 contains ~7.3M parameters compared to ResNet50 (~25M), reducing overfitting risk on medium-sized cohorts.

---

## 🛡️ 2. The Critical BatchNorm Policy During Transfer Learning

### Problem
During Stage 2 fine-tuning on domain-specific radiographs, the batch size is often small to medium (e.g. 32–64 per replica). If Batch Normalization layers in the backbone remain trainable, the moving average mean $\mu$ and variance $\sigma^2$ are updated by noisy batch statistics, corrupting ImageNet channel calibrations and causing catastrophic forgetting.

### Solution
In `unfreeze_densenet_for_finetuning`:
```python
for layer in backbone.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
```
All Batch Normalization layers were strictly locked, keeping moving statistics frozen while allowing convolutional kernels in the top 20 layers to adapt to radiograph textures.

---

## 🎯 3. Decision Threshold Optimization Without Test Leakage

### Metric Trade-off
Under the default 0.50 threshold on the 4,003-patient test set:
- Sensitivity: $73.17\%$
- Specificity: $78.97\%$
- False Positives: $652$

By running an 81-point threshold scan **exclusively on validation predictions**, the optimal threshold of **$t = 0.60$** was chosen to maximize validation F1. Applied to the held-out test set, this yielded:
- Specificity: **$84.17\%$** (**161 fewer false alarms**, false positives dropped from 652 to 491)
- Accuracy: **$79.79\%$** ($+2.12\%$ increase)
- ROC-AUC: **$0.8381$** (Invariant global separability)
- PR-AUC: **$0.6022$** (Invariant precision/recall trade-off)
