# Phase 32 — Production Deployment & Infrastructure Report

**Project:** Skin Lesion Classification  
**Phase:** 32 — Production Deployment & Infrastructure Engineering  
**Status:** COMPLETE  
**Production Checkpoint:** `models/production/resnet50_skin_lesion_v1.keras`  
**Model SHA-256 Checksum:** `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e`  

> **EXPLICIT STATEMENT:**  
> "No model training or model modification was performed during Phase 32. The production checkpoint remains frozen and byte-for-byte immutable."

---

## 1. Deployment Architecture

```text
User Browser / API Client
          │
          ▼  (HTTPS Port 443)
┌─────────────────────────────────────────┐
│ NGINX / Cloudflare Reverse Proxy        │
│ - TLS Termination                       │
│ - Static Asset Cache                    │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
┌──────────────────┐ ┌───────────────────────────────┐
│ Static Frontend  │ │ FastAPI ASGI Backend Container│
│ (React + Vite)   │ │ (Docker - Uvicorn Port 8000)  │
└──────────────────┘ └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Frozen ResNet50 Model         │
                     │ (resnet50_skin_lesion_v1.keras)│
                     └───────────────────────────────┘
```

---

## 2. Backend Container Deployment

* **Dockerfile Spec (`backend/Dockerfile`):**
  * Base Image: `python:3.11-slim`
  * User Security: Non-root user `appuser` (UID 1000)
  * Container Port: `8000`
  * Healthcheck: `HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/api/v1/health || exit 1`
  * Command: `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1`

* **Container Build & Launch Commands:**
  ```bash
  # Build production image
  docker build -t dermavision-backend:v1 -f backend/Dockerfile .

  # Run container
  docker run -d --name dermavision-api -p 8000:8000 dermavision-backend:v1
  ```

---

## 3. Frontend Production Build & Hosting

* **Build Specification:**
  ```bash
  cd frontend
  npm run build
  ```
* **Output:** Static HTML/JS/CSS assets emitted to `frontend/dist/`.
* **Hosting Strategy:** Compatible with NGINX, AWS S3 + CloudFront, Vercel, or Netlify. Single Page Application (SPA) routing fallback configured for `index.html`.

---

## 4. Environment Variables Matrix

| Variable Name | Purpose | Development Default | Production Recommended |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | Application mode | `development` | `production` |
| `LOG_LEVEL` | Logging verbosity | `INFO` | `WARNING` / `INFO` |
| `API_HOST` | FastAPI bind IP | `0.0.0.0` | `0.0.0.0` |
| `API_PORT` | FastAPI bind port | `8000` | `8000` |
| `MODEL_PATH` | Keras model path | `models/production/...` | `models/production/...` |
| `FRONTEND_ORIGIN` | Allowed CORS origin | `http://localhost:5173` | `https://dermavision.app` |
| `VITE_API_BASE_URL` | Client API target | `http://localhost:8000` | `https://api.dermavision.app` |

---

## 5. Model Packaging & Versioning

* **Checkpoint Location:** `models/production/resnet50_skin_lesion_v1.keras`
* **Model Checksum:** `93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e`
* **Version Policy:** Production model versions follow explicit semantic tags (`resnet50_skin_lesion_v1`, `v2`, etc.). Future versions require SHA-256 manifest registration before deployment.

---

## 6. Security Hardening

1. **Security Headers Middleware:** Sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, and `X-XSS-Protection`.
2. **Path Traversal Protection:** Input filenames sanitized via `Path(filename).name`.
3. **Strict Upload Restrictions:** Whitelist extension check (`.jpg`, `.jpeg`, `.png`, `.webp`), MIME type check, PIL binary header decoding check, and 10MB file size limit.
4. **CORS:** Restricted to `FRONTEND_ORIGIN`. Wildcard origins (`*`) disabled in production.

---

## 7. Health & Readiness Semantics

* `GET /api/v1/health`
  * **Liveness:** `"alive"`
  * **Readiness:** `"ready"` (Returns HTTP 200 OK when model is loaded into memory; returns HTTP 533/503 if model initialization failed).

---

## 8. Scaling & Memory Considerations

Deep learning models require significant memory (~200MB - 1GB GPU/CPU RAM per instance). Running multiple Uvicorn worker processes (`--workers 4`) duplicates the Keras model in each process memory space.

**Production Recommendation:** Use a single-worker Uvicorn process per container instance (`--workers 1`) and scale horizontally by deploying multiple container replicas behind a load balancer (e.g. NGINX or AWS ALB).

---

## 9. Rollback & Recovery Procedures

If a deployment failure or corrupted deployment occurs:
1. Re-deploy previous Docker container image (`dermavision-backend:v1-previous`).
2. Verify production model SHA-256 against `models/production/model_sha256.txt`.
3. Restart container and confirm `GET /api/v1/health` returns HTTP 200 `readiness: ready`.

---

## 10. Phase 32 Completion Status

* **Model Checksum Integrity:** **VERIFIED (100% Match)**
* **Dockerfile & Dockerignore:** **CREATED & TESTED**
* **Health/Readiness Semantics:** **IMPLEMENTED**
* **Process Restart Simulation:** **PASSED 100%**
* **Phase 31 QA Regression Suite:** **13 / 13 PASSED**
* **System Status:** **DEPLOYMENT PACKAGE COMPLETE & PRODUCTION READY**
