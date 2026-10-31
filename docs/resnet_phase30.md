# Phase 30 — Grad-CAM Explainability Implementation Report

**Project:** Skin Lesion Classification  
**Phase:** 30 — Production Grad-CAM Explainability Integration  
**Status:** COMPLETE  
**Production Model Checkpoint:** `models/production/resnet50_skin_lesion_v1.keras`  
**Model SHA-256 Checksum:** `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e`  

> **EXPLICIT STATEMENT:**  
> "No model training or model modification was performed during Phase 30. The production checkpoint remains frozen."

---

## 1. Objective

Phase 30 integrates a production-grade Gradient-weighted Class Activation Mapping (Grad-CAM) explainability engine into the skin lesion classification system. Grad-CAM visualizes spatial feature activation maps over lesion images, highlighting the exact image regions that statistically contributed to the frozen ResNet50 model's top prediction.

---

## 2. Architecture & Data Flow

```
User Selects Image (React Frontend)
          │
          ▼
   POST /api/v1/predict ──► Returns Top-3 Predictions JSON
          │
          ▼
   POST /api/v1/explain ──► Triggers Grad-CAM Engine
          │
          ▼
Frozen ResNet50 Model (models/production/resnet50_skin_lesion_v1.keras)
          │
          ▼
Target Layer Gradients (conv5_block3_out)
          │
          ▼
Heatmap Generation & Bilinear Resizing & Jet Colormap Overlay
          │
          ▼
Base64 Data-URL Packaging (Original, Heatmap, Overlay)
          │
          ▼
React ExplainabilityPanel (Interactive Overlay/Heatmap/Original Tabs)
```

* **Grad-CAM Generator (`src/gradcam.py`):** Wraps the frozen Keras submodel (`resnet50`) to track activations and gradients at the final residual block (`conv5_block3_out`).
* **Explainability Service (`backend/explainability.py`):** Service layer providing thread-safe image processing and Data-URL encoding.
* **REST API (`POST /api/v1/explain`):** FastAPI endpoint validating file payloads and returning structured explainability responses.
* **React Component (`ExplainabilityPanel.jsx`):** Interactive segmented view controller (`Overlay`, `Heatmap`, `Original`) with non-blocking error handling and prominent safety disclaimers.

---

## 3. Discovered Target Layer

Through programmatic inspection of `models/production/resnet50_skin_lesion_v1.keras`, the final convolutional layer of the ResNet50 backbone was identified:
* **Submodel:** `resnet50`
* **Target Layer Name:** `conv5_block3_out`
* **Output Shape:** `(None, 7, 7, 2048)`

---

## 4. Grad-CAM Mathematical Method

1. **Forward Pass:** Image batch $(1, 224, 224, 3)$ is passed through the model to extract feature maps $A^k \in \mathbb{R}^{7 \times 7}$ at `conv5_block3_out` and logit score $y^c$ for target class $c$.
2. **Gradient Computation:** Gradients $\frac{\partial y^c}{\partial A^k}$ are calculated via `tf.GradientTape()`.
3. **Global Average Pooling Weights:** Channel weights $\alpha_k^c$ are computed:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
4. **ReLU Activation Map:** Weighted linear combination of feature maps is activated by ReLU to isolate positive contributions:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
5. **Normalization & Colormap:** Heatmap is normalized to $[0, 1]$, resized via bilinear interpolation to original image dimensions, colored via Matplotlib `jet` colormap, and blended with original RGB pixels ($\alpha = 0.5$).

---

## 5. API Response Contract (`POST /api/v1/explain`)

```json
{
  "success": true,
  "prediction": {
    "class_name": "Squamous cell carcinoma",
    "class_index": 12,
    "confidence": 0.3417
  },
  "explainability": {
    "method": "Grad-CAM",
    "target_class": "Squamous cell carcinoma",
    "target_class_index": 12,
    "layer": "conv5_block3_out"
  },
  "images": {
    "original": "data:image/jpeg;base64,...",
    "heatmap": "data:image/png;base64,...",
    "overlay": "data:image/jpeg;base64,..."
  },
  "latency_ms": 1164.34,
  "disclaimer": "Grad-CAM visualization highlights image regions that contributed to the model's prediction. It is provided for AI explainability and research purposes and is NOT a medical diagnosis."
}
```

---

## 6. Safety & Non-Diagnostic Disclaimer

Grad-CAM output is strictly positioned as **statistical feature attention visualization**:
> "Grad-CAM visualization highlights image regions that contributed to the model prediction. This visualization shows AI feature attention for research purposes and is NOT a medical diagnosis or proof of disease."

---

## 7. Model Integrity Check

Model SHA-256 checksum was verified before and after execution:
* **Expected Checksum:** `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e`
* **Actual Checksum:** `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e`
* **Status:** VERIFIED UNTOUCHED & FROZEN.

---

## 8. Test Suite Verification (`backend/test_explainability.py`)

All integration test cases passed cleanly:
1. `Model SHA-256 Integrity Verification` → PASS
2. `POST /api/v1/explain` (Valid Non-Test Image) → PASS
3. `POST /api/v1/explain` (Invalid `.txt` Extension) → PASS (HTTP 415)
4. `POST /api/v1/explain` (Unsupported `.bmp` Format) → PASS (HTTP 415)
5. `POST /api/v1/explain` (Corrupted `.jpg` Binary) → PASS (HTTP 400)
6. `POST /api/v1/explain` (Empty 0-byte Upload) → PASS (HTTP 400)
7. `Frontend Production Build (npm run build)` → PASS (Built in 155 ms)

---

## 9. Next Phase

* **Phase 31:** Production QA, Security & Performance Hardening.
