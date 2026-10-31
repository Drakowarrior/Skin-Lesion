# Backend Startup, Environment Configuration & DEF-03 Resolution

## 1. Executive Summary & Root Cause Analysis of DEF-03

### Root Cause of DEF-03
In previous audits, **DEF-03** stated:
> *"The background production daemon on port 8001 may require an operator restart or reload to apply updated `.env` settings."*

An in-depth investigation of `backend/config.py` and `backend/main.py` established the exact mechanics:
1. **Import-Time Environment Binding:** `backend/config.py` loads environment variables using `python-dotenv` (`load_dotenv(PROJECT_ROOT / "backend" / ".env")`) at **import time**. Key security and operational parameters (such as `ALLOWED_ORIGINS`, `METRICS_AUTH_TOKEN`, `MODEL_ENGINE`, and `MAX_UPLOAD_SIZE`) are evaluated once when module imports take place and assigned to module-level constants.
2. **Process Isolation in Python/Uvicorn:** When a server daemon runs (via `uvicorn backend.main:app`), it retains its in-memory module state. Modifying the on-disk `.env` file does **not** trigger a hot reload unless Uvicorn is executed with `--reload` or the daemon is restarted.
3. **Port Alignment in Frontend Client:** In `frontend/src/services/api.js`, the fallback API base URL was historically defaulted to `http://127.0.0.1:8000`. When `VITE_API_BASE_URL` was not passed, it fell back to port 8000 instead of port 8001. We hardened `frontend/src/services/api.js` to default strictly to `http://127.0.0.1:8001`, matching `frontend/.env` and the intended production backend port.
4. **Conclusion:** DEF-03 is **NOT** a bug in code or process failure. It is the expected, safe operational behavior of standard Python application runtimes. When starting the backend for an academic demo, simply launching Uvicorn binds the full, updated `.env` configuration immediately.

---

## 2. Configuration Settings Audit Matrix

| Configuration Variable | Required / Optional | Present in `.env`? | Validated Behavior | Consuming Module | Restart Required? |
| :--- | :---: | :---: | :--- | :--- | :---: |
| `ENVIRONMENT` | Required | Yes (`production`) | Enables production mode and triggers fail-safe CORS logic. | `backend/config.py`, `backend/main.py` | Yes |
| `API_HOST` | Optional | Yes (`0.0.0.0`) | Specifies network binding interface. For local demo, loopback `127.0.0.1` is recommended. | `backend/config.py` | Yes |
| `API_PORT` | Required | Yes (`8001`) | Specifies listening port. Matches frontend configuration. | `backend/config.py` | Yes |
| `MODEL_ENGINE` | Required | Yes (`convnext`) | Selects active champion model ConvNeXt-Small vs rollback ResNet50. | `backend/config.py`, `backend/main.py` | Yes |
| `MODEL_PATH` | Required | Yes (`models/production/...`) | Points to verified PyTorch checkpoint file on disk. | `backend/config.py`, `backend/main.py` | Yes |
| `MAX_UPLOAD_SIZE` | Required | Yes (`15728640`) | Enforces 15 MB ($15,728,640$ bytes) upload limit before PIL decoding. | `backend/config.py`, `backend/main.py` | Yes |
| `ALLOWED_ORIGINS` | Required in prod | Yes (`http://localhost:5173,...`) | Restricts cross-origin requests to local frontend. If unset in prod, defaults to empty `[]`. | `backend/config.py`, `backend/main.py` | Yes |
| `METRICS_AUTH_TOKEN`| Required for metrics | Yes (Configured) | Protects `/metrics` with Bearer token authentication (401/403/200). | `backend/observability.py` | Yes |
| `AI_PROVIDER` | Optional | Yes (`groq`) | LLM provider for educational guidance assistance. | `backend/guidance.py` | Yes |
| `AI_MODEL` | Optional | Yes (`qwen/qwen3.8-27b`)| Target model for educational triage guidance. | `backend/guidance.py` | Yes |
| `AI_API_KEY` | Optional | Yes (Configured) | Server-side API key for guidance generation. Auto-falls back to local rules if offline. | `backend/guidance.py` | Yes |

*Note: In accordance with project security constraints, all sensitive values and tokens are omitted.*

---

## 3. Verified Backend Startup Command

The authoritative, verified startup procedure for the FastAPI backend is:

```bash
cd "/Users/karangarg/Documents/coding/skin lesions"
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

### Verification Checks:
- **Virtual Environment:** Python 3.11.15 in `.venv`.
- **Working Directory:** Repository root (`/Users/karangarg/Documents/coding/skin lesions`).
- **Loopback Binding:** `--host 127.0.0.1` binds securely to localhost, preventing unwanted external network exposure.
- **Port:** `8001`, aligning directly with `frontend/.env` and `frontend/src/services/api.js`.

---

## 4. Health Check Verification (`/api/v1/health`)

Executing `GET http://127.0.0.1:8001/api/v1/health` returns HTTP 200 OK:
```json
{
  "status": "healthy",
  "liveness": "alive",
  "readiness": "ready",
  "model": "convnext_small_v1",
  "model_version": "v1",
  "architecture": "ConvNeXt-Small",
  "model_loaded": true,
  "explainability_loaded": true,
  "uptime_seconds": 0.0
}
```

### Observations:
- **Liveness:** `alive`
- **Readiness:** `ready`
- **Active Model:** `convnext_small_v1` (ConvNeXt-Small)
- **Explainability:** `true` (Grad-CAM loaded and hooked into `features.7`)
- **Secrets:** Zero private credentials, API keys, or system paths exposed.

---

## 5. Port-Conflict Handling & Safe Process Management

If port 8001 is already occupied when starting the service:

1. **Check Occupying Process:**
   ```bash
   lsof -i :8001
   ```
2. **Inspect Process Details:**
   ```bash
   ps -f -p <PID>
   ```
3. **If it is the intended Uvicorn process and healthy:**
   Verify `curl -s http://127.0.0.1:8001/api/v1/health`. If healthy, the server is already active and ready for demonstration.
4. **If a restart is required to load new configuration:**
   Terminate only that specific PID gracefully:
   ```bash
   kill -TERM <PID>
   ```
   Never use wildcard `pkill` or kill unrelated user services.

