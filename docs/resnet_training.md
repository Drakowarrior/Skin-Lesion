# ResNet50 Improved Training & Transfer Learning Pipeline (Phase 15)

## 1. Overview & Baseline Context

- **Dataset**: 14 Skin Condition Classes (ISIC 2019/HAM10000 + MSLDv2.0)
- **Baseline Test Results (Phase 13)**:
  - Top-1 Accuracy: 73.27%
  - Top-3 Accuracy: 94.18%
  - Macro F1: 69.65%
  - Weighted F1: 74.66%
- **Baseline Checkpoint**: `models/resnet50_best.keras` (Preserved intact for comparison)
- **Target Improved Checkpoint**: `models/resnet50_improved_best.keras`

---

## 2. Key Improvements Implemented

### A. Data Augmentation
- **Strategy**: Medical-image tailored transformations avoiding unnatural distortion.
- **Layers**:
  - `RandomFlip("horizontal_and_vertical")`
  - `RandomRotation(0.10)` (Small rotation)
  - `RandomZoom(0.10)` (Small zoom)
  - `RandomTranslation(0.08, 0.08)` (Small translation)
  - `RandomContrast(0.08)` (Mild contrast adjustments)

### B. Class Imbalance Strategy
- **Problem**: Extreme raw class weights (e.g. Dermatofibroma weight ~10.97) caused overconfident false positive predictions.
- **Solution**: Power-scaled balanced weighting (`weights^0.6`) clipped between `[0.25, 5.0]`.
- **Justification**: Maintains effective gradient compensation for minority classes without gradient explosion or extreme probability distortions.

### C. Refined Classification Head
- **Architecture**:
  ```
  Input (224, 224, 3)
  ↓
  ResNet50 Backbone
  ↓
  GlobalAveragePooling2D
  ↓
  BatchNormalization
  ↓
  Dense(256, activation="relu")
  ↓
  Dropout(0.3)
  ↓
  Dense(14, activation="softmax")
  ```

### D. Fine-Tuning & BatchNorm Safety
- **Stage 1 (`head`)**: Backbone frozen (`base_model.trainable = False`), Adam LR `1e-3`.
- **Stage 2 (`finetune`)**: Unfreeze top 30 ResNet layers, Adam LR `1e-5`.
- **BatchNorm Safety**: All `BatchNormalization` layers explicitly kept non-trainable during fine-tuning to preserve ImageNet statistics stability.

---

## 3. Terminal Execution Instructions

### Validation Command (Non-Training):
```bash
source .venv/bin/activate
python -m src.resnet_train --validate-only
```

### Stage 1 Training Command (Head Only):
```bash
source .venv/bin/activate
python -m src.resnet_train --stage head --epochs 10 --batch-size 32
```

### Stage 2 Training Command (Fine-Tuning):
```bash
source .venv/bin/activate
python -m src.resnet_train --stage finetune --epochs 15 --batch-size 32
```
