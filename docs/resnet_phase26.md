The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.
# Phase 26 — Final Controlled Boundary Optimization Report

## Executive Summary

Phase 26 represents the **FINAL planned training experiment** for the ResNet50 skin lesion classification project.
Phase 26 initialized directly from the **Phase 25 checkpoint** (`models/resnet50_phase25_best.keras`) and applied conservative fine-tuning ($	ext{lr} = 2	ext{e-6}$, top 30 layers trainable, all 54 BN layers frozen, $\lambda = 0.01$) with model selection strictly driven by **Validation Macro F1** (`mode="max"`).

---

## 1. Experimental Controls & Unchanged Variables

* **Backbone**: ResNet50 Pretrained + Phase 25 Fine-Tuned Weights
* **Classification Head**: GAP + BN + Dense(256, relu) + Dropout(0.3) + Dense(14, softmax)
* **Dataset Splits**: Train (25,649), Val (5,478), Test (5,529 - 100% Isolated)
* **Oversampling**: STRICTLY DISABLED (`oversample = False`)
* **Class Weight Strategy**: Power-scaled ($w^{0.60}$, clip $[0.40, 4.00]$)
* **Model Selection Metric**: Validation Macro F1 (`mode="max"`)

---

## 2. Eight-Way Master Benchmarking Table (Untouched Test Split: N = 5,529)

| Metric | Phase 13 Baseline | Phase 16 Improved | Phase 17 Rebalanced | Phase 20 Baseline | Phase 23 LDAM | Phase 24 Pairwise | Phase 25 Candidate | Phase 26 Final | Net Diff (P26 vs P25) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Top-1 Accuracy** | 73.27% | 74.32% | 72.4% | 77.46% | 75.24% | 75.73% | 79.76% | **79.96%** | +0.20% |
| **Top-1 Correct Count** | 4,051 | 4,109 | 4,003 | 4,283 | 4,160 | 4,187 | 4,410 | **4,421** | +11 images |
| **Top-3 Accuracy** | 94.18% | 95.06% | 94.63% | 95.42% | 94.9% | 95.51% | 95.98% | **96.42%** | +0.44% |
| **Macro Precision** | 68.33% | 67.66% | 66.59% | 72.17% | 68.12% | 69.6% | 75.71% | **76.23%** | +0.52% |
| **Macro Recall** | 73.68% | 68.71% | 69.77% | 70.97% | 68.67% | 69.76% | 73.05% | **75.37%** | +2.32% |
| **Macro F1 Score** | 69.65% | 67.79% | 67.09% | 71.4% | 67.91% | 68.41% | 74.18% | **75.49%** | **+1.31%** |
| **Weighted Precision** | 77.28% | 75.93% | 75.44% | 77.22% | 75.26% | 76.32% | 79.11% | **80.34%** | +1.23% |
| **Weighted Recall** | 73.27% | 74.32% | 72.4% | 77.46% | 75.24% | 75.73% | 79.76% | **79.96%** | +0.20% |
| **Weighted F1 Score** | 74.66% | 74.84% | 73.19% | 77.22% | 75.01% | 75.64% | 79.23% | **80.07%** | +0.84% |
| **Expected Calibration Error (ECE)** | 4.15% | 3.96% | 10.6% | 4.93% | 4.2% | 6.41% | 6.16% | **8.7%** | +2.54% |
| **Brier Score** | 0.357 | 0.35029 | 0.39217 | 0.31597 | 0.34577 | 0.33484 | 0.29465 | **0.30405** | +0.00940 |

---

## 3. Specific Target Verification: Melanocytic nevi ↔ Melanoma

| Metric / Check | Phase 25 Candidate | Phase 26 Boundary Optimized | Net Difference |
| :--- | ---: | ---: | :---: |
| **Nevi $\rightarrow$ Melanoma Errors** | 73 | **122** | +49 errors |
| **Melanoma $\rightarrow$ Nevi Errors** | 210 | **166** | -44 errors |
| **Total Mutual Nevi $\leftrightarrow$ Melanoma Confusion** | 283 | **288** | **+5 errors** |
| **Melanocytic nevi Recall** | 90.57% | **86.42%** | -4.15% |
| **Melanocytic nevi F1** | 85.71% | **86.16%** | +0.45% |
| **Melanoma Recall** | 54.20% | **59.50%** | +5.30% |
| **Melanoma F1** | 58.46% | **60.34%** | +1.88% |

---

*Report generated automatically by Phase 26 Evaluation Pipeline.*