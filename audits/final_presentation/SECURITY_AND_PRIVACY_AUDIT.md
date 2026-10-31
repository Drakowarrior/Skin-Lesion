# Security and Privacy Audit Report

## 1. Security Architecture & Controls

This audit examines the network security, access control, upload defense, and privacy protections implemented in the backend and frontend services.

---

## 2. Verification Matrix

| Security Control | Implementation Details | Verification Evidence | Status |
| :--- | :--- | :--- | :---: |
| **CORS Fail-Safe Allowlist** | When `ENVIRONMENT=production` and `ALLOWED_ORIGINS` is unset, the origin list strictly defaults to empty `[]`. Explicit trusted origins (`http://localhost:5173,http://127.0.0.1:5173`) are enforced when configured. | Verified via `scripts/test_phase22_security.py` (Test `SEC-22-01`). | **PASS** |
| **Prometheus Metrics Protection**| `/metrics` requires `Authorization: Bearer <METRICS_AUTH_TOKEN>`. Unconfigured returns 403; missing header returns 401 with `WWW-Authenticate`; invalid token returns 403; valid token exports metrics with HTTP 200. | Verified via `scripts/test_phase22_security.py` (Tests `SEC-22-02` through `SEC-22-05`). | **PASS** |
| **15 MB Upload Size Defense** | Strictly enforced at 15 MB ($15,728,640$ bytes) across backend `config.py`, API routes, and client file picker. Oversized files rejected before decoding. | Source code inspection across backend and frontend. | **PASS** |
| **Path Traversal Defense** | File uploads are read directly into memory via `io.BytesIO` and validated by Pillow. No unvalidated user filenames are written to disk. | Source code inspection of `backend/main.py`. | **PASS** |
| **Zero Secret Exposure** | API keys and tokens are loaded strictly from environment variables (`.env`). No secrets are committed to tracking or printed in reports. | Automated scans over source files and markdown deliverables. | **PASS** |
| **Patient Privacy Safeguard**| Uploaded lesion images are processed in-memory for classification and Grad-CAM. Raw images are never sent to external LLM providers. | Verified in `backend/guidance.py` and `backend/main.py`. | **PASS** |
| **Rollback Checkpoint Integrity**| Frozen ResNet50 baseline checkpoint (`models/production/resnet50_skin_lesion_v1.keras`) is verified and intact for instant rollback. | Verified via SHA-256 calculation (`93a121...d58e`). | **PASS** |

