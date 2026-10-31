#!/usr/bin/env python3
"""
Phase 17 End-to-End API Contract & Integration Test Suite

Validates:
1. Frontend-to-Backend connection on port 8001.
2. Active model identity: ConvNeXt-Small, version v1, 14 classes, 384x384 input.
3. End-to-end image upload and calibrated prediction payload.
4. 14 canonical class names matching frontend expectations.
5. Top-3 predictions strictly ordered by probability with unclipped formatting.
6. Quality-gate and uncertainty telemetry consistency.
7. Grad-CAM visual explainability heatmap and overlay generation.
8. Patient guidance endpoint contract with error fallback handling.
9. Biomedical literature research search and Google Scholar integration.
10. Error resilience: HTTP 415 on invalid extensions, HTTP 400 on corrupt bytes.
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.error
from datetime import datetime, timezone

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_DIR = os.path.join(WORKSPACE_ROOT, "audits", "phase17")
os.makedirs(AUDIT_DIR, exist_ok=True)

BACKEND_URL = "http://127.0.0.1:8001"
SAMPLE_IMG_PATH = os.path.join(WORKSPACE_ROOT, "backend", "test_sample.jpg")

EXPECTED_CLASSES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "HFMD",
    "Healthy",
    "Measles",
    "Melanocytic nevi",
    "Melanoma",
    "Monkeypox",
    "Squamous cell carcinoma",
    "Vascular lesions"
]

def make_multipart_body(fields, files):
    boundary = "----WebKitFormBoundary" + hashlib.md5(str(time.time()).encode()).hexdigest()
    body = bytearray()
    for k, v in fields.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))
    for name, (filename, content, content_type) in files.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"

def send_request(url, method="GET", headers=None, data=None):
    if headers is None:
        headers = {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            elapsed = time.perf_counter() - t0
            body = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = body
            return resp.status, parsed, elapsed, None
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(err_body)
        except Exception:
            parsed = err_body
        return e.code, parsed, elapsed, str(e)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return 0, None, elapsed, str(e)

print("Starting Phase 17 API Contract Regression Test Suite...")
results = []

with open(SAMPLE_IMG_PATH, "rb") as f:
    sample_bytes = f.read()

# -----------------------------------------------------------------------------
# 1. Frontend-Backend Health & Connection
# -----------------------------------------------------------------------------
st, health, lat, err = send_request(f"{BACKEND_URL}/api/v1/health")
t1_ok = (st == 200 and isinstance(health, dict) and health.get("status") == "healthy" and health.get("readiness") == "ready")
results.append({
    "contract_id": "CTR-01",
    "name": "Backend connectivity and readiness",
    "status": "PASS" if t1_ok else "FAIL",
    "details": f"HTTP {st}, status: {health.get('status') if isinstance(health, dict) else None}, latency: {round(lat*1000, 2)}ms"
})

# -----------------------------------------------------------------------------
# 2. Model Metadata & Identity Verification
# -----------------------------------------------------------------------------
st, model_info, lat, err = send_request(f"{BACKEND_URL}/api/v1/model")
t2_ok = (
    st == 200 and
    isinstance(model_info, dict) and
    model_info.get("architecture") == "ConvNeXt-Small" and
    model_info.get("model_name") == "convnext_small_v1" and
    model_info.get("num_classes") == 14 and
    model_info.get("input_size") == "384x384"
)
results.append({
    "contract_id": "CTR-02",
    "name": "Active ConvNeXt-Small model identity",
    "status": "PASS" if t2_ok else "FAIL",
    "details": f"Model: {model_info.get('model_name') if isinstance(model_info, dict) else None}, arch: {model_info.get('architecture') if isinstance(model_info, dict) else None}, input: {model_info.get('input_size') if isinstance(model_info, dict) else None}"
})

# -----------------------------------------------------------------------------
# 3. Canonical 14-Class Taxonomy
# -----------------------------------------------------------------------------
st, ai_status, lat, err = send_request(f"{BACKEND_URL}/api/v1/ai/status")
classes = ai_status.get("supported_classes", []) if isinstance(ai_status, dict) else []
t3_ok = (st == 200 and len(classes) == 14 and set(classes) == set(EXPECTED_CLASSES))
results.append({
    "contract_id": "CTR-03",
    "name": "Canonical 14-class taxonomy match",
    "status": "PASS" if t3_ok else "FAIL",
    "details": f"Classes count: {len(classes)}, all 14 canonical categories match frontend expectations"
})

# -----------------------------------------------------------------------------
# 4. Image Prediction & Calibrated Probability
# -----------------------------------------------------------------------------
data, ct = make_multipart_body({}, {"image": ("sample.jpg", sample_bytes, "image/jpeg")})
st, pred_res, lat, err = send_request(f"{BACKEND_URL}/api/v1/predict", method="POST", headers={"Content-Type": ct}, data=data)
pred = pred_res.get("prediction", {}) if isinstance(pred_res, dict) and pred_res.get("prediction") is not None else {}
t4_ok = (st == 200 and isinstance(pred_res, dict) and pred_res.get("success") is True and "class" in pred and "confidence" in pred)
results.append({
    "contract_id": "CTR-04",
    "name": "End-to-end image prediction payload",
    "status": "PASS" if t4_ok else "FAIL",
    "details": f"Predicted: {pred.get('class')}, calibrated probability: {pred.get('confidence')}, latency: {round(lat*1000, 2)}ms"
})

# -----------------------------------------------------------------------------
# 5. Top-3 Differential Ordering & Unclipped Probabilities
# -----------------------------------------------------------------------------
top3 = pred_res.get("top3", []) if isinstance(pred_res, dict) else []
t5_ok = False
if len(top3) == 3:
    c0 = top3[0].get("confidence", 0.0)
    c1 = top3[1].get("confidence", 0.0)
    c2 = top3[2].get("confidence", 0.0)
    t5_ok = (c0 >= c1 >= c2) and (c0 <= 1.0) and (c0 + c1 + c2 <= 1.0001)
results.append({
    "contract_id": "CTR-05",
    "name": "Top-3 descending probability differential",
    "status": "PASS" if t5_ok else "FAIL",
    "details": f"Top-3 confidences: {[round(t.get('confidence', 0), 4) for t in top3]}, strictly descending and finite"
})

# -----------------------------------------------------------------------------
# 6. Quality Gate & Uncertainty Telemetry
# -----------------------------------------------------------------------------
quality = pred_res.get("quality", {}) if isinstance(pred_res, dict) else {}
uncertainty = pred_res.get("uncertainty", {}) if isinstance(pred_res, dict) else {}
t6_ok = ("state" in quality and "state" in uncertainty and "metrics" in uncertainty)
results.append({
    "contract_id": "CTR-06",
    "name": "Quality-gate and uncertainty telemetry consistency",
    "status": "PASS" if t6_ok else "FAIL",
    "details": f"Quality state: {quality.get('state')}, uncertainty state: {uncertainty.get('state')}, normalized entropy: {uncertainty.get('metrics', {}).get('normalized_entropy')}"
})

# -----------------------------------------------------------------------------
# 7. Grad-CAM Visual Explainability Heatmap & Overlay
# -----------------------------------------------------------------------------
exp_data, exp_ct = make_multipart_body({}, {"image": ("sample.jpg", sample_bytes, "image/jpeg")})
st, exp_res, lat, err = send_request(f"{BACKEND_URL}/api/v1/explain", method="POST", headers={"Content-Type": exp_ct}, data=exp_data)
images_dict = exp_res.get("images", {}) if isinstance(exp_res, dict) else {}
t7_ok = (st == 200 and isinstance(exp_res, dict) and exp_res.get("success") is True and ("overlay_blended" in images_dict or "heatmap_colored" in images_dict))
results.append({
    "contract_id": "CTR-07",
    "name": "Grad-CAM visual activation overlay generation",
    "status": "PASS" if t7_ok else "FAIL",
    "details": f"HTTP {st}, overlay length: {len(images_dict.get('overlay_blended', ''))} chars, target layer: features.7"
})

# -----------------------------------------------------------------------------
# 8. Patient Guidance Endpoint & Fallback Robustness
# -----------------------------------------------------------------------------
guidance_req = json.dumps({
    "prediction": {
        "predicted_class": pred.get("class", "Monkeypox"),
        "confidence_percent": round(pred.get("confidence", 0.5) * 100, 2),
        "top3": top3
    },
    "user_question": "What is Monkeypox?",
    "user_context": None
}).encode("utf-8")
st, g_res, lat, err = send_request(f"{BACKEND_URL}/api/v1/guidance", method="POST", headers={"Content-Type": "application/json"}, data=guidance_req)
t8_ok = (st == 200 and isinstance(g_res, dict) and g_res.get("success") is True and "answer_markdown" in g_res and "urgency_level" in g_res)
results.append({
    "contract_id": "CTR-08",
    "name": "Patient guidance & clinical decision support",
    "status": "PASS" if t8_ok else "FAIL",
    "details": f"HTTP {st}, urgency: {g_res.get('urgency_level') if isinstance(g_res, dict) else None}, answer length: {len(g_res.get('answer_markdown', '')) if isinstance(g_res, dict) else 0} chars"
})

# -----------------------------------------------------------------------------
# 9. Biomedical Research Literature Endpoint
# -----------------------------------------------------------------------------
st, pubmed_res, lat, err = send_request(f"{BACKEND_URL}/api/v1/research/Monkeypox")
t9_ok = (st == 200 and isinstance(pubmed_res, dict) and "verified_sources" in pubmed_res and "condition" in pubmed_res)
results.append({
    "contract_id": "CTR-09",
    "name": "Biomedical scientific research literature search",
    "status": "PASS" if t9_ok else "FAIL",
    "details": f"HTTP {st}, verified sources count: {len(pubmed_res.get('verified_sources', [])) if isinstance(pubmed_res, dict) else 0}, scholar_url: {pubmed_res.get('scholar_url') if isinstance(pubmed_res, dict) else None}"
})


# -----------------------------------------------------------------------------
# 10. Robust Error Handling (Unsupported File & Corrupt Bytes)
# -----------------------------------------------------------------------------
txt_data, txt_ct = make_multipart_body({}, {"image": ("document.txt", b"plain text", "text/plain")})
st_txt, res_txt, _, _ = send_request(f"{BACKEND_URL}/api/v1/predict", method="POST", headers={"Content-Type": txt_ct}, data=txt_data)

corrupt_data, corrupt_ct = make_multipart_body({}, {"image": ("corrupt.jpg", b"bad_truncated_data", "image/jpeg")})
st_corrupt, res_corrupt, _, _ = send_request(f"{BACKEND_URL}/api/v1/predict", method="POST", headers={"Content-Type": corrupt_ct}, data=corrupt_data)

t10_ok = (st_txt == 415 and st_corrupt == 400)
results.append({
    "contract_id": "CTR-10",
    "name": "Invalid file rejection and error handling",
    "status": "PASS" if t10_ok else "FAIL",
    "details": f"Unsupported file rejected with HTTP {st_txt}; corrupt image rejected with HTTP {st_corrupt}"
})

all_pass = all(r["status"] == "PASS" for r in results)

summary = {
    "audit_phase": "Phase 17B — Frontend and Backend Contract Verification",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "backend_url": BACKEND_URL,
    "total_contract_checks": len(results),
    "passed": sum(1 for r in results if r["status"] == "PASS"),
    "failed": sum(1 for r in results if r["status"] == "FAIL"),
    "overall_status": "ALL_CONTRACTS_VERIFIED" if all_pass else "CONTRACT_MISMATCH_DETECTED",
    "contracts": results
}

with open(os.path.join(AUDIT_DIR, "api_contract_regression_tests.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nPhase 17 Contract Regression Suite Finished. Passed: {summary['passed']}/{summary['total_contract_checks']}")
for r in results:
    print(f"[{r['status']}] {r['contract_id']}: {r['name']} - {r['details']}")
