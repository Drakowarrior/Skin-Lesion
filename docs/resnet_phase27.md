The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.
# Phase 27 — Production Model Freeze

## 1. Objective

Phase 27 transitions the project from iterative model experimentation into production software engineering.
The primary goal is to **freeze** the high-performing Phase 26 ResNet50 model checkpoint (`models/resnet50_phase26_best.keras`), establish a versioned production model artifact (`models/production/resnet50_skin_lesion_v1.keras`), verify model integrity via SHA-256 checksums, and define a clean, reusable Python inference API (`SkinLesionPredictor`) for future backend and UI integration.

> **Crucial Rule**: Phase 27 does not introduce another training experiment. The Phase 26 checkpoint is frozen as the production candidate and subsequent work focuses on inference engineering, API integration, frontend development, testing, explainability, and deployment.

---

## 2. Why Training Stops Here

Across 26 iterations of controlled experimental progression, the ResNet50 pipeline achieved substantial performance gains over the baseline:

| Iteration | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 | Brier Score | ECE | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Phase 13 Baseline** | 73.27% | 94.18% | 69.65% | 74.66% | 0.35700 | 4.15% | Superseded |
| **Phase 16 Improved** | 74.32% | 95.06% | 67.79% | 74.84% | 0.35029 | 3.96% | Superseded |
| **Phase 17 Rebalanced** | 72.40% | 94.63% | 67.09% | 73.19% | 0.39217 | 10.60% | Rejected |
| **Phase 20 Baseline** | 77.46% | 95.42% | 71.40% | 77.22% | 0.31597 | 4.93% | Superseded |
| **Phase 23 LDAM** | 75.24% | 94.90% | 67.91% | 75.01% | 0.34577 | 4.20% | Rejected |
| **Phase 24 Pairwise** | 75.73% | 95.51% | 68.41% | 75.64% | 0.33484 | 6.41% | Rejected |
| **Phase 25 Candidate** | 79.76% | 95.98% | 74.18% | 79.23% | 0.29465 | 6.16% | Candidate |
| **Phase 26 Final** | **79.96%** | **96.42%** | **75.49%** | **80.07%** | **0.30405** | **8.70%** | **FROZEN CHAMPION** |

Phase 26 achieved the highest overall Top-1 accuracy (**79.96%**), Top-3 accuracy (**96.42%**), Macro F1 (**75.49%**), and Weighted F1 (**80.07%**).
Continued loss function tweaking or minor hyperparameter adjustments yield diminishing returns and risk overfitting. Therefore, model training is officially **concluded**.

---

## 3. Frozen Model Artifacts

* **Source Checkpoint**: `models/resnet50_phase26_best.keras`
* **Production Model Location**: `models/production/resnet50_skin_lesion_v1.keras`
* **Integrity Checksum File**: `models/production/model_sha256.txt`
* **Model Metadata File**: `models/production/model_metadata.json`

---

## 4. Model Architecture

* **Backbone**: ResNet50 ImageNet Pretrained + Phase 26 Fine-Tuned Weights
* **Input Dimensions**: `(224, 224, 3)`
* **Classification Head**:
  - `GlobalAveragePooling2D`
  - `BatchNormalization` (Frozen)
  - `Dense(256, activation='relu')`
  - `Dropout(rate=0.3)`
  - `Dense(14, activation='softmax')`
* **Output Dimension**: `14` Softmax Probabilities

---

## 5. Canonical Class Mapping Contract

Defined strictly in `src/class_labels.py`:

| Index | Label | Category |
| :---: | :--- | :--- |
| `0` | Actinic keratoses | Premalignant / Malignant |
| `1` | Basal cell carcinoma | Malignant |
| `2` | Benign keratosis-like lesions | Benign |
| `3` | Chickenpox | Infectious |
| `4` | Cowpox | Infectious |
| `5` | Dermatofibroma | Benign |
| `6` | HFMD | Infectious |
| `7` | Healthy | Normal Skin |
| `8` | Measles | Infectious |
| `9` | Melanocytic nevi | Benign |
| `10` | Melanoma | Malignant |
| `11` | Monkeypox | Infectious |
| `12` | Squamous cell carcinoma | Malignant |
| `13` | Vascular lesions | Benign |

---

## 6. Input Preprocessing Pipeline

The production predictor (`src/inference.py`) applies the exact Caffe-style ResNet50 preprocessing pipeline:
1. Loads image (RGB, RGBA, Grayscale 'L', or Palette 'P').
2. Respects EXIF orientation tags and converts to 3-channel RGB.
3. Resizes image to `(224, 224)` via bilinear interpolation.
4. Converts pixel values to `float32` in $[0.0, 255.0]$.
5. Applies `tf.keras.applications.resnet50.preprocess_input`:
   - Converts RGB to BGR channel ordering.
   - Subtracts ImageNet mean values `[103.939, 116.779, 123.68]`.

---

## 7. Production Inference Contract

The Python inference API (`SkinLesionPredictor.predict()`) returns a structured dictionary:

```json
{
  "predicted_class": "Melanocytic nevi",
  "predicted_index": 9,
  "confidence": 0.8745,
  "probability_sum": 1.000000,
  "probabilities": {
    "Actinic keratoses": 0.0012,
    "Melanocytic nevi": 0.8745,
    "Melanoma": 0.0812,
    ...
  },
  "top3": [
    {"class": "Melanocytic nevi", "index": 9, "confidence": 0.8745},
    {"class": "Melanoma", "index": 10, "confidence": 0.0812},
    {"class": "Benign keratosis-like lesions", "index": 2, "confidence": 0.0211}
  ]
}
```

---

## 8. Model Integrity Verification

Model integrity is verified via SHA-256 hash stored in `models/production/model_sha256.txt`.
Any accidental alteration or corruption of `resnet50_skin_lesion_v1.keras` during deployment can be detected immediately by comparing the runtime SHA-256 digest.

---

## 9. Test-Set Isolation

The test split (`data/processed/test.csv`, 5,529 images) was:
- **NEVER** used for model weight updates or training.
- **NEVER** used for hyperparameter, threshold, or calibration tuning.
- **ONLY** evaluated in read-only mode during final phase benchmarking.

---

## 10. Clinical & System Limitations

1. **Statistical Model Confidence**: Predicted probabilities represent statistical softmax model confidence scores, **NOT** medical certainty, clinical diagnostic accuracy, or safety guarantees.
2. **Calibration Notice**: Phase 26 achieved an Expected Calibration Error (ECE) of **8.70%**. Outputs should be displayed as "model predicted confidence" without implying perfect clinical calibration.
3. **Research & Educational Scope**: This system is designed solely as a decision support research artifact and demonstration tool, **NOT** as a standalone medical diagnostic device.

---

## 11. Next Engineering Roadmap

With model training concluded and Phase 27 complete, the project transitions into product software development:

- **Phase 28**: Backend / API Development (REST API endpoints for prediction)
- **Phase 29**: Frontend UI Development (Interactive Web Interface)
- **Phase 30**: Explainability / Grad-CAM Integration (Visual heatmap generation)
- **Phase 31**: Integration Testing & Edge Case Validation
- **Phase 32**: Performance Optimization & Error Handling
- **Phase 33**: Deployment Pipeline
- **Phase 34**: Final Documentation & Presentation