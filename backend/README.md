# Skin Lesion Classification — FastAPI Backend (Phase 32 Deployment Package)

Production REST API service for the frozen ResNet50 skin lesion classification model and Grad-CAM explainability engine.

---

## 📌 Features & Deployment Specification

- **Docker Containerization:** Configured with production `Dockerfile` (non-root `appuser`, HEALTHCHECK probe, 1 worker).
- **Liveness & Readiness Probes:** `GET /api/v1/health` returns liveness and readiness status (HTTP 503 if model uninitialized).
- **Security Headers & CORS:** Enforces `X-Frame-Options`, `X-Content-Type-Options`, and environment-driven `ALLOWED_ORIGINS`.
- **Model Checksum Integrity:** Frozen model SHA-256 (`93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e`).

---

## 🚀 Local Development Server Command

```bash
# Activate virtualenv and start FastAPI on port 8000
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker Deployment Commands

```bash
# Build production backend image
docker build -t dermavision-backend:v1 -f backend/Dockerfile .

# Run container
docker run -d --name dermavision-api -p 8000:8000 dermavision-backend:v1
```

---

## 🧪 Testing & QA Commands

```bash
# Process Restart & Deployment Test
python -m backend.test_deployment

# Full QA, Security & Performance Test Suite
python -m backend.test_qa_suite
```

