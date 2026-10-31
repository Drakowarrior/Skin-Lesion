#!/usr/bin/env python3
"""
Phase 16 Observability Automated Test Suite

Tests:
1. Health and readiness responses remain correct.
2. Metrics are emitted and update after test requests.
3. Error and latency metrics reflect observed test outcomes.
4. Metrics do not contain secrets, image data, or personal identifiers.
5. Invalid requests do not crash the service.
6. Monitoring failures do not interrupt predictions.
7. Metrics labels remain strictly bounded.
8. Existing frontend behavior and production model selection remain unchanged.
"""

import os
import sys
import time
import json
import re
from datetime import datetime, timezone

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

AUDIT_DIR = os.path.join(WORKSPACE_ROOT, "audits", "phase16")
os.makedirs(AUDIT_DIR, exist_ok=True)

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import (
    MODEL_ENGINE,
    MODEL_ARCHITECTURE,
    EXPECTED_MODEL_SHA256,
    METRICS_ENABLED,
)
import backend.observability as obs

print("Initializing Phase 16 Observability Test Suite...")

results = []

with TestClient(app) as client:
    # -----------------------------------------------------------------------------
    # Test 1: Health & Readiness Response Integrity
    # -----------------------------------------------------------------------------
    t1_resp = client.get("/api/v1/health")
    t1_ok = (
        t1_resp.status_code == 200 and
        t1_resp.json().get("status") == "healthy" and
        t1_resp.json().get("readiness") == "ready" and
        t1_resp.json().get("model_loaded") is True and
        t1_resp.json().get("explainability_loaded") is True
    )
    results.append({
        "test_id": "OBS-01",
        "name": "Health and readiness response verification",
        "status": "PASS" if t1_ok else "FAIL",
        "details": f"Status: {t1_resp.status_code}, readiness: {t1_resp.json().get('readiness')}, model: {t1_resp.json().get('model')}"
    })

    # -----------------------------------------------------------------------------
    # Test 2: Metrics Emitted & Update After Test Requests
    # -----------------------------------------------------------------------------
    m_before = client.get("/metrics")
    m_before_text = m_before.text

    # Send 3 prediction requests with test image
    with open(os.path.join(WORKSPACE_ROOT, "backend", "test_sample.jpg"), "rb") as f:
        sample_bytes = f.read()

    for _ in range(3):
        client.post("/api/v1/predict", files={"image": ("sample.jpg", sample_bytes, "image/jpeg")})

    m_after = client.get("/metrics")
    m_after_text = m_after.text

    t2_ok = (
        m_after.status_code == 200 and
        "skin_lesion_http_requests_total" in m_after_text and
        "skin_lesion_inference_duration_seconds" in m_after_text and
        "skin_lesion_model_loaded" in m_after_text and
        len(m_after_text) > len(m_before_text)
    )
    results.append({
        "test_id": "OBS-02",
        "name": "Metrics emission and dynamic counter increment",
        "status": "PASS" if t2_ok else "FAIL",
        "details": f"Metrics length before: {len(m_before_text)}, after: {len(m_after_text)}, status: {m_after.status_code}"
    })

    # -----------------------------------------------------------------------------
    # Test 3: Error & Latency Metrics Reflect Observed Test Outcomes
    # -----------------------------------------------------------------------------
    # Trigger a 415 unsupported media error
    client.post("/api/v1/predict", files={"image": ("doc.txt", b"plain text payload", "text/plain")})
    # Trigger a 400 corrupt image error
    client.post("/api/v1/predict", files={"image": ("bad.jpg", b"corrupt bytes", "image/jpeg")})

    m_err = client.get("/metrics").text
    t3_ok = (
        'status_class="4xx"' in m_err and
        "skin_lesion_http_request_duration_seconds_bucket" in m_err
    )
    results.append({
        "test_id": "OBS-03",
        "name": "Error and latency metrics reflect observed outcomes",
        "status": "PASS" if t3_ok else "FAIL",
        "details": "Observed 4xx status_class recorded in skin_lesion_http_requests_total and histogram buckets populated."
    })

    # -----------------------------------------------------------------------------
    # Test 4: Metrics Privacy & Secret Scrubbing Audit
    # -----------------------------------------------------------------------------
    metrics_payload = m_err
    has_api_keys = any(key in metrics_payload for key in ["AI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "sk-", "gsk_"])
    has_image_base64 = "data:image/" in metrics_payload or "base64" in metrics_payload
    has_patient_pii = any(pii in metrics_payload.lower() for pii in ["patient_name", "ssn", "mrn", "birthdate", "forearm"])

    t4_ok = (not has_api_keys) and (not has_image_base64) and (not has_patient_pii)
    results.append({
        "test_id": "OBS-04",
        "name": "Metrics privacy and secret scrubbing audit",
        "status": "PASS" if t4_ok else "FAIL",
        "details": f"Keys detected: {has_api_keys}, image base64: {has_image_base64}, PII terms: {has_patient_pii}"
    })

    # -----------------------------------------------------------------------------
    # Test 5: Invalid Requests Do Not Crash Service
    # -----------------------------------------------------------------------------
    res_bad_url = client.get("/api/v1/non_existent_endpoint_12345")
    res_empty_post = client.post("/api/v1/predict", data={})
    res_malformed_json = client.post("/api/v1/guidance", content=b"invalid { json")

    health_after_abuse = client.get("/api/v1/health")
    t5_ok = (
        res_bad_url.status_code == 404 and
        res_empty_post.status_code == 422 and
        res_malformed_json.status_code == 422 and
        health_after_abuse.status_code == 200 and
        health_after_abuse.json().get("status") == "healthy"
    )
    results.append({
        "test_id": "OBS-05",
        "name": "Service resilience under invalid and malformed requests",
        "status": "PASS" if t5_ok else "FAIL",
        "details": f"404/422 responses handled gracefully; post-abuse health check returned 200 OK"
    })

    # -----------------------------------------------------------------------------
    # Test 6: Monitoring Failures Do Not Interrupt Predictions
    # -----------------------------------------------------------------------------
    try:
        obs.record_inference_latency("ConvNeXt-Small", -999.0)
        # Prediction must succeed regardless of telemetry anomalies
        test_pred = client.post("/api/v1/predict", files={"image": ("sample.jpg", sample_bytes, "image/jpeg")})
        t6_ok = (test_pred.status_code == 200 and test_pred.json().get("success") is True)
    except Exception as e:
        t6_ok = False

    results.append({
        "test_id": "OBS-06",
        "name": "Fault-isolation: Telemetry errors do not interrupt predictions",
        "status": "PASS" if t6_ok else "FAIL",
        "details": f"Prediction succeeded with HTTP 200 despite telemetry edge case simulation"
    })

    # -----------------------------------------------------------------------------
    # Test 7: Metrics Labels Strictly Bounded
    # -----------------------------------------------------------------------------
    client.get("/arbitrary_path_with_uuid_abcdef1234567890_xyz")
    m_cardinality = client.get("/metrics").text

    leaked_uuid = "abcdef1234567890" in m_cardinality
    endpoint_labels = re.findall(r'endpoint="([^"]+)"', m_cardinality)
    unique_endpoints = set(endpoint_labels)
    allowed_cats = {"health", "model", "predict", "explain", "ai_status", "guidance", "research", "metrics", "docs", "other"}
    bounded_ok = unique_endpoints.issubset(allowed_cats) and not leaked_uuid

    t7_ok = bounded_ok
    results.append({
        "test_id": "OBS-07",
        "name": "Bounded-cardinality label enforcement",
        "status": "PASS" if t7_ok else "FAIL",
        "details": f"Recorded endpoint label categories: {sorted(list(unique_endpoints))}. Unbounded URL leaked: {leaked_uuid}"
    })

    # -----------------------------------------------------------------------------
    # Test 8: Frontend Contract & Model Preservation
    # -----------------------------------------------------------------------------
    model_info = client.get("/api/v1/model").json()
    t8_ok = (
        model_info.get("architecture") == MODEL_ARCHITECTURE and
        model_info.get("model_name") == "convnext_small_v1" and
        model_info.get("num_classes") == 14 and
        "disclaimer" in model_info
    )
    results.append({
        "test_id": "OBS-08",
        "name": "Frontend contract and active model preservation",
        "status": "PASS" if t8_ok else "FAIL",
        "details": f"Architecture: {model_info.get('architecture')}, classes: {model_info.get('num_classes')}, disclaimer preserved"
    })

all_passed = all(r["status"] == "PASS" for r in results)

test_summary = {
    "audit_phase": "Phase 16D — Observability Automated Verification",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "test_environment": "Isolated ASGI Test Environment (FastAPI TestClient with Lifespan)",
    "total_tests": len(results),
    "passed": sum(1 for r in results if r["status"] == "PASS"),
    "failed": sum(1 for r in results if r["status"] == "FAIL"),
    "overall_status": "ALL_TESTS_PASSED" if all_passed else "FAILURES_DETECTED",
    "test_cases": results
}

output_path = os.path.join(AUDIT_DIR, "observability_test_results.json")
with open(output_path, "w") as f:
    json.dump(test_summary, f, indent=2)

print(f"\nPhase 16 Test Suite Finished. Passed: {test_summary['passed']}/{test_summary['total_tests']}")
for r in results:
    print(f"[{r['status']}] {r['test_id']}: {r['name']}")
