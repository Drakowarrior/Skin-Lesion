# Phase 28 — FastAPI Backend Report

**Project:** Skin Lesion Classification  
**Phase:** 28 — FastAPI Backend & REST API Engineering  
**Status:** COMPLETE  
**Production Model Checkpoint:** `models/production/resnet50_skin_lesion_v1.keras`  

> **EXPLICIT STATEMENT:**  
> "Phase 28 does not perform model training. It exposes the frozen Phase 27 production candidate through a controlled API."

---

## 1. Objective

The primary objective of Phase 28 is to transition the project from ML model development into production application engineering. A clean, resilient FastAPI backend was constructed to expose the frozen ResNet50 classifier via standardized RESTful HTTP APIs for consumption by web (React) and mobile frontends.

---

## 2. Backend Architecture

The backend follows clean application design principles, decoupling API route handlers, configuration management, and inference logic:

```
React Frontend / API Client
          │
          ▼
POST /api/v1/predict  │  GET /api/v1/health  │  GET /api/v1/model
          │
          ▼
    FastAPI Router (backend/main.py)
          │
          ▼
  SkinLesionPredictor (src/inference.py)
          │
          ▼
Frozen ResNet50 Model (models/production/resnet50_skin_lesion_v1.keras)
          │
          ▼
  Structured Prediction JSON Response
```

* **Module Breakdown:**
  * `backend/main.py`: Main FastAPI application, router definitions, CORS middleware, lifespan events, and error handling.
  * `backend/config.py`: Environment configuration loader supporting relative paths, upload limits, and CORS origins.
  * `backend/requirements.txt`: Lightweight production dependencies (`fastapi`, `uvicorn`, `pydantic`, `python-multipart`, `httpx`).
  * `src/inference.py`: Canonical `SkinLesionPredictor` wrapper handling PIL loading, RGB normalization, Caffe preprocessing, and top-3 formatting.
  * `src/class_labels.py`: Single source of truth for 14 canonical skin condition labels.

---

## 3. API Endpoints

| Method | Endpoint | Description | Response Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health check endpoint returning operational and model readiness status | `200 OK` |
| `GET` | `/api/v1/model` | Model metadata endpoint exposing frozen architecture details and 14 classes | `200 OK` |
| `POST` | `/api/v1/predict` | Multipart image prediction endpoint returning top-3 predicted skin conditions | `200 OK`, `400 Bad Request`, `413 Too Large`, `415 Unsupported` |

---

## 4. Model Loading

* **Lifespan Initialization:** The model is initialized exactly once during server startup inside FastAPI's `@asynccontextmanager` lifespan handler and bound to `app.state.predictor`.
* **Zero-Re-Loading:** Requests reuse the in-memory `SkinLesionPredictor` instance, avoiding costly per-request model loading or file I/O operations.
* **Read-Only Guarantee:** Model compilation is disabled (`compile=False`), ensuring zero weight modification or retraining.

---

## 5. Image Validation

Uploaded files undergo strict multi-stage validation prior to inference execution:
1. **Presence Check:** Ensures file payload is attached and filename is non-empty.
2. **Format/Extension Verification:** Validates file extension against permitted formats (`.jpg`, `.jpeg`, `.png`, `.webp`). Rejects unsupported formats (e.g. `.bmp`, `.gif`, `.txt`) with `HTTP 415`.
3. **Payload Size Guard:** Enforces `MAX_UPLOAD_SIZE` (default 10 MB). Oversized requests are rejected with `HTTP 413` before reading full memory buffer.
4. **Header & Integrity Decoding:** Uses `PIL.Image.open()` and `verify()` to catch corrupted binaries, truncated files, or non-image files with spoofed extensions.

---

## 6. Prediction Response Schema

Successful predictions return standard JSON structured as follows:

```json
{
  "success": true,
  "prediction": {
    "class": "Melanoma",
    "class_index": 10,
    "confidence": 0.3951
  },
  "top3": [
    { "class": "Melanoma", "class_index": 10, "confidence": 0.3951 },
    { "class": "Squamous cell carcinoma", "class_index": 12, "confidence": 0.3417 },
    { "class": "Melanocytic nevi", "class_index": 9, "confidence": 0.1515 }
  ],
  "model": {
    "name": "resnet50_skin_lesion_v1",
    "version": "v1"
  },
  "latency_ms": 1535.29,
  "disclaimer": "This tool provides an AI model prediction for research/educational purposes and is not a medical diagnosis."
}
```

---

## 7. Error Handling

Custom exception handlers format all errors as clean JSON objects, hiding raw Python tracebacks, internal filesystem paths, or stack details:

```json
{
  "detail": "Corrupt or invalid image file. Unable to decode image contents."
}
```

---

## 8. CORS Configuration

CORS middleware (`fastapi.middleware.cors.CORSMiddleware`) is configured to allow cross-origin requests from the React frontend development server (`FRONTEND_ORIGIN`, default `http://localhost:5173`). Wildcard `allow_origins=["*"]` is disabled for security compliance.

---

## 9. Testing & Validation

Integration tests in `backend/test_api.py` verify all endpoints against empirical requirements using FastAPI `TestClient`:

* **Test Isolation:** Strictly isolates the test dataset (`data/processed/test.csv`). Smoke tests exclusively utilize training image samples (`data/raw/train/...`).
* **Test Results:**
  1. `GET /api/v1/health` → PASS
  2. `GET /api/v1/model` → PASS
  3. `POST /api/v1/predict` (Valid Non-Test Image) → PASS
  4. `POST /api/v1/predict` (Invalid `.txt` Extension) → PASS (HTTP 415)
  5. `POST /api/v1/predict` (Unsupported `.bmp` Format) → PASS (HTTP 415)
  6. `POST /api/v1/predict` (Corrupted `.jpg` Binary) → PASS (HTTP 400)
  7. `POST /api/v1/predict` (Empty 0-byte Upload) → PASS (HTTP 400)

---

## 10. Performance Smoke Test

* **Health Endpoint Latency:** ~1.55 ms
* **Model Endpoint Latency:** ~0.57 ms
* **Cold/First Prediction Latency (Metal GPU compilation):** ~1535.29 ms
* **Invalid Payload Rejection Latency:** ~0.52 ms – 0.92 ms

---

## 11. Security Considerations

* No execution of arbitrary user files.
* Strict image binary decoding checks eliminate image-based polyglot/payload exploits.
* Hides internal directory trees and Python tracebacks.
* Relative paths avoid leaking local system environment details.

---

## 12. Medical Disclaimer

"This tool provides an AI model prediction for research/educational purposes and is not a medical diagnosis. Consult a board-certified dermatologist for medical evaluation."

---

## 13. Next Phase

* **Phase 29:** React Frontend Development & Integration with FastAPI Backend.
