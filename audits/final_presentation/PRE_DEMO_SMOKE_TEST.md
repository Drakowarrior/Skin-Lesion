# Final Pre-Demo Live Runtime Smoke Test Report

## 1. Executive Summary & Verification Context

This report records the live end-to-end runtime verification of the **Skin Lesion Classification & Explainable AI System** executed directly against the active services:
- **FastAPI Backend:** Active on `http://127.0.0.1:8001` (PID 90434, uptime $> 3500\text{s}$)
- **Vite React Frontend:** Active on `http://localhost:5173` (PID 90643, Vite 8.3.2)
- **Active Model Checkpoint:** `models/production/convnext_small_skin_lesion_v1.pt` (SHA-256: `67f4d8...fe3`)
- **Rollback Model Checkpoint:** `models/production/resnet50_skin_lesion_v1.keras` (SHA-256: `93a121...d58e`)

---

## 2. Live Runtime & Network Request Matrix

| Pipeline Step / Endpoint | Verification Mechanism | Observed Response / Output | Status |
| :--- | :--- | :--- | :---: |
| **Backend Health Check** | `GET http://127.0.0.1:8001/api/v1/health` | HTTP 200 OK. `status: "healthy"`, `model_loaded: true`, `architecture: "ConvNeXt-Small"`. | **PASS** |
| **Model Metadata Query** | `GET http://127.0.0.1:8001/api/v1/model` | HTTP 200 OK. 14 classes, $384 \times 384$ resolution, `convnext_small_v1`. | **PASS** |
| **Frontend HTTP Readiness**| `GET http://localhost:5173/` | HTTP 200 OK. HTML rendered, React bundle scripts injected cleanly. | **PASS** |
| **Frontend Status Indicator**| Headless Chrome DOM inspection | Navbar badge displays `<div class="status-badge-minimal online"><span class="status-text">Service Ready</span></div>`. | **PASS** |
| **Live Image Inference** | `POST http://127.0.0.1:8001/api/v1/predict` (`test_sample.jpg`) | HTTP 200 OK. Class: **Monkeypox**, Confidence: `0.4902`, Prob sum: `1.0000`, Quality: `QUALITY_PASS`. | **PASS** |
| **Top-3 Ordering & Values**| `POST http://127.0.0.1:8001/api/v1/predict` | Top 3 correctly sorted descending: `['Monkeypox', 'Cowpox', 'Measles']`. Probabilities finite in $[0, 1]$. | **PASS** |
| **Live Grad-CAM Generation**| `POST http://127.0.0.1:8001/api/v1/explain` (`test_sample.jpg`) | HTTP 200 OK. Target layer: `features.7 (ConvNeXtBlock Stage 4)`. Generated valid base64 Data URLs. | **PASS** |
| **Grad-CAM Opacity 0%** | Visual style inspection | Heatmap style `opacity: 0`. Raw clinical photograph 100% visible beneath without chromatic tint. | **PASS** |
| **Grad-CAM Opacity 40%** | Visual style inspection | Balanced alpha-blend (`opacity: 0.40`). High-activation features highlight salient pigment networks. | **PASS** |
| **Grad-CAM Opacity 100%** | Visual style inspection | Full salience activation map (`opacity: 1.00`). No blank panels, clipping, or layer dropouts. | **PASS** |
| **Spatial Alignment** | Dual-layer DOM container | Both `.base-layer` and `.heatmap-overlay-layer` share `position: absolute; object-fit: contain;` inside 380px wrapper. | **PASS** |
| **Invalid Upload Rejection** | `POST http://127.0.0.1:8001/api/v1/predict` (non-image payload) | HTTP 400 Bad Request. Clean JSON detail: `Unsupported file extension '.py'`. Zero synthetic fallback. | **PASS** |
| **Literature Separation** | UI inspection | PubMed citations appear strictly inside patient guidance cards; Google Scholar appears in exactly 1 external card. | **PASS** |
| **Screenshot Evidence** | Headless Chrome screenshot capture | Saved desktop (`live_app_desktop.png`, 138 KB) and mobile (`live_app_mobile.png`, 68 KB) screenshots. | **PASS** |

---

## 3. Environment & Configuration Lifecycle (DEF-03 Verification)

- **Configuration Loading:** `backend/config.py` loads environment variables using `python-dotenv` at **module import time**.
- **Process Binding:** Settings (`ALLOWED_ORIGINS`, `METRICS_AUTH_TOKEN`, `MODEL_ENGINE`, `MAX_UPLOAD_SIZE`) are evaluated once on startup. Modifying `.env` on disk requires restarting the daemon.
- **Port Alignment:** `frontend/src/services/api.js` defaults cleanly to `http://127.0.0.1:8001`, matching the live backend service port.
- **Security Validation:**
  - Production CORS allowlist fail-safe defaults to empty `[]` if unset in production.
  - Prometheus `/metrics` Bearer token authentication verified (401 for missing token, 403 for invalid token, 200 for valid token).
  - Maximum upload size is enforced at $15\text{ MB}$ ($15,728,640$ bytes).
  - Zero secrets or private keys are printed in logs or public responses.

---

## 4. Regression Test Execution Summary

| Test Domain | Target / Command | Status | Details |
| :--- | :--- | :---: | :--- |
| **Backend Unit Tests** | `KERAS_HOME=/tmp/.keras MPLCONFIGDIR=/tmp/.matplotlib .venv/bin/python -m unittest discover backend` | **PASS** | **23 of 23 unit tests passed** in 12.850s |
| **Security Hardening** | `.venv/bin/python scripts/test_phase22_security.py` | **PASS** | **5 of 5 tests passed** |
| **Frontend Production Build** | `cd frontend && npm run build` | **PASS** | Vite built production bundle in **207 ms** (0 errors) |
| **Frontend Static Linter** | `cd frontend && npx oxlint` | **PASS** | **0 errors**, 15 non-blocking warnings |
| **Live Prediction API** | `POST http://127.0.0.1:8001/api/v1/predict` | **PASS** | Validated live against active background process PID 90434 |
| **Live Grad-CAM API** | `POST http://127.0.0.1:8001/api/v1/explain` | **PASS** | Validated live against active background process PID 90434 |

---

## 5. Checkpoint Integrity Confirmation

Both checkpoints were verified before and after all testing operations and remain bit-for-bit unchanged:

- **Active ConvNeXt-Small Checkpoint (`models/production/convnext_small_skin_lesion_v1.pt`):**  
  `67f4d8d04edef9389aeac77688dcc90e193b717de67f8c4c0c1ebf15483a3fe3` (**MATCH**)
- **Rollback ResNet50 Checkpoint (`models/production/resnet50_skin_lesion_v1.keras`):**  
  `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e` (**MATCH**)
