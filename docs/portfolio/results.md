# 📊 MedVision-AI: Verified Results & Benchmark Audit

> **Comprehensive breakdown of primary held-out test performance and threshold optimization impact.**

---

## 🏆 Final Held-Out Test Split Benchmarks (4,003 Unique Patients)

| Metric | Frozen Threshold ($t=0.60$) | Reference ($t=0.50$) | Delta | Significance |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC (Primary)** | **`0.8381`** | `0.8380` | `+0.0001` | High global separability across all operating rates |
| **PR-AUC (Primary)** | **`0.6022`** | `0.6023` | `-0.0001` | High precision/recall trade-off under 22.5% prevalence |
| **Specificity** | **`84.17%`** | `78.97%` | **`+5.20%`** | **161 fewer false alarms** (FP reduced from 652 to 491) |
| **Accuracy** | **`79.79%`** | `77.67%` | **`+2.12%`** | Overall classification accuracy |
| **Precision** | **`54.33%`** | `50.30%` | **`+4.03%`** | Higher positive predictive value |
| **Sensitivity** | **`64.75%`** | `73.17%` | `-8.42%` | True positive detection rate (584 / 902 confirmed cases) |
| **F1-Score** | **`0.5908`** | `0.5962` | `-0.0054` | Balanced harmonic trade-off |

---

### Confusion Matrix on Held-Out Test Set ($N = 4,003$ Patients @ Frozen $t = 0.60$):
- **True Positives (TP):** `584`
- **True Negatives (TN):** `2,610`
- **False Positives (FP):** `491`
- **False Negatives (FN):** `318`
