# Phase 17 — Minority-Class Performance Improvement & Targeted Rebalancing

## 1. Motivation

In Phase 16, the ResNet50 model achieved high overall Top-1 (74.32%) and Top-3 (95.06%) accuracy. However, performance on small underrepresented minority classes suffered (e.g. Vascular lesions F1 dropped to 66.67%, Dermatofibroma F1 to 31.17%, Squamous cell carcinoma to 28.96%).

Phase 17 addresses minority-class representation using a dual targeted rebalancing approach:
1. **Smoothed Power-Scaled Class Weighting**
2. **Targeted Class-Aware Minority Oversampling** (capped at $3.0\times$)

---

## 2. Dataset Analysis (Train & Val Only)

Analysis was performed strictly on `train.csv` (25,649 images) and `val.csv` (5,478 images). The test set (`test.csv`, 5,529 images) remained **100% untouched and isolated**.

- **Total Train Images**: 25,649
- **Total Val Images**: 5,478
- **Minimum Class**: Dermatofibroma (167 train samples, 35 val samples)
- **Maximum Class**: Melanocytic nevi (9,016 train samples, 1,929 val samples)
- **Imbalance Ratio**: 53.99 : 1

---

## 3. Targeted Rebalancing Strategy

### A. Class-Weight Formula & Smoothing
```
effective_count = count + smoothing (20)
raw_weight = (median_class_count / effective_count) ^ 0.75
weight = clip(raw_weight, 0.25, 7.0)
```
- **Smoothing (+20)**: Prevents tiny sample noise from causing runaway weight spikes.
- **Power (0.75)**: Provides stronger gradient compensation than Phase 16 (0.60) without the instability of un-capped linear inverse weighting.
- **Cap [0.25, 7.0]**: Protects training stability.

### B. Targeted Minority Oversampling
- Applied **strictly to training dataset (`train.csv`)**.
- Caps minority class sample duplication to a **$3.0\times$ multiplier** (e.g., Dermatofibroma increases from 167 to 501 training samples).
- **Zero Synthetic Images**: No SMOTE, GANs, MixUp, or CutMix. Duplicated index pointers undergo random medical image augmentation during dataset loading.
- **Validation and Test splits are NEVER oversampled**.

---

## 4. Model Architecture & Checkpoint Isolation

- **Backbone**: `ResNet50` (ImageNet pretrained weights)
- **Head**: `GlobalAveragePooling2D` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dense(256, relu)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(14, softmax)`
- **BatchNorm Handling**: All 54 `BatchNormalization` layers explicitly set to `trainable=False` during Stage 2 fine-tuning.
- **Checkpoint Isolation**:
  - `models/resnet50_best.keras` (Phase 13 Baseline: **Untouched**)
  - `models/resnet50_improved_best.keras` (Phase 16 Model: **Untouched**)
  - `models/resnet50_phase17_best.keras` (**New Phase 17 Checkpoint**)

---

## 5. Terminal Execution Instructions

### Validation Command (Non-Training):
```bash
source .venv/bin/activate
python -m src.resnet_phase17_train --validate-only
```

### Stage 1 Training Command (Head Only):
```bash
source .venv/bin/activate
python -m src.resnet_phase17_train --stage head --epochs 10 --batch-size 32
```

### Stage 2 Training Command (Fine-Tuning):
```bash
source .venv/bin/activate
python -m src.resnet_phase17_train --stage finetune --epochs 15 --batch-size 32
```
