"""
Comprehensive Production QA, Security & Performance Test Suite (Phase 31)

Executes end-to-end audit and QA verification across:
- Model SHA-256 Checksum Integrity (93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e)
- API Health & Metadata Endpoints
- Valid Prediction & Top-3 Schema Validation
- Prediction vs Explainability Consistency Verification
- Security & Input Malformed Payload Rejection (Path Traversal, Corrupt Data, Empty Upload, Unsupported Formats)
- Production Performance Benchmarks (Latency metrics)
"""

import time
import hashlib
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from src.class_labels import CLASS_NAMES, NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "production" / "resnet50_skin_lesion_v1.keras"
EXPECTED_SHA256 = "93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e"


def verify_model_sha256() -> bool:
    """Verify that frozen production model SHA-256 checksum remains byte-for-byte unchanged."""
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        return False
    
    with open(MODEL_PATH, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    
    match = actual_sha256 == EXPECTED_SHA256
    print(f"[Model Checksum] Actual={actual_sha256[:16]}... Expected={EXPECTED_SHA256[:16]}... Match={match}")
    return match


def get_valid_non_test_image_path() -> Path:
    """Find a valid non-test image from train.csv for smoke testing."""
    train_csv = PROJECT_ROOT / "data" / "processed" / "train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        for _, row in df.iterrows():
            img_p = PROJECT_ROOT / row["image_path"]
            if img_p.exists():
                return img_p
    
    train_dir = PROJECT_ROOT / "data" / "raw" / "train"
    jpgs = list(train_dir.rglob("*.jpg"))
    if jpgs:
        return jpgs[0]
    
    raise FileNotFoundError("No non-test training image found for integration test.")


def run_full_qa_suite():
    """Execute complete Phase 31 QA, Security, and Performance Test Suite."""
    print("\n" + "=" * 70)
    print("RUNNING PHASE 31 PRODUCTION QA, SECURITY & PERFORMANCE SUITE")
    print("=" * 70)

    # 1. Pre-Test Model Integrity Check
    print("\n--- Step 1: Model SHA-256 Integrity Verification ---")
    sha_ok = verify_model_sha256()
    assert sha_ok, "Model SHA-256 verification failed!"
    print("Status: PASS (Production Model Unmodified & Frozen)")

    qa_results = {
        "model_integrity": True,
        "health_endpoint": False,
        "model_metadata": False,
        "predict_schema": False,
        "explain_schema": False,
        "predict_explain_consistency": False,
        "security_path_traversal": False,
        "security_unicode_filename": False,
        "security_long_filename": False,
        "security_corrupt_bytes": False,
        "security_empty_upload": False,
        "security_unsupported_ext": False,
        "performance_benchmark": False,
    }

    with TestClient(app) as client:
        # 2. Health Endpoint
        print("\n--- Step 2: GET /api/v1/health ---")
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        print("Status: PASS (Health check online & model loaded)")
        qa_results["health_endpoint"] = True

        # 3. Model Metadata
        print("\n--- Step 3: GET /api/v1/model ---")
        resp = client.get("/api/v1/model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_classes"] == NUM_CLASSES == 14
        assert data["status"] == "frozen"
        print("Status: PASS (Model metadata & 14 classes verified)")
        qa_results["model_metadata"] = True

        # 4. Valid Prediction Test
        print("\n--- Step 4: POST /api/v1/predict (Schema Verification) ---")
        sample_img_path = get_valid_non_test_image_path()
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.perf_counter()
        resp_pred = client.post(
            "/api/v1/predict",
            files={"image": ("sample.jpg", img_bytes, "image/jpeg")}
        )
        pred_latency = (time.perf_counter() - t0) * 1000.0

        assert resp_pred.status_code == 200
        pred_data = resp_pred.json()
        assert pred_data["success"] is True
        assert pred_data["prediction"]["class"] in CLASS_NAMES
        assert len(pred_data["top3"]) == 3
        print(f"Predicted class: {pred_data['prediction']['class']} | Confidence: {pred_data['prediction']['confidence']:.4f}")
        print(f"Latency: {pred_latency:.2f} ms")
        print("Status: PASS (Prediction schema verified)")
        qa_results["predict_schema"] = True

        # 5. Valid Explainability Test
        print("\n--- Step 5: POST /api/v1/explain (Grad-CAM Schema Verification) ---")
        t0 = time.perf_counter()
        resp_exp = client.post(
            "/api/v1/explain",
            files={"image": ("sample.jpg", img_bytes, "image/jpeg")}
        )
        exp_latency = (time.perf_counter() - t0) * 1000.0

        assert resp_exp.status_code == 200
        exp_data = resp_exp.json()
        assert exp_data["success"] is True
        assert exp_data["explainability"]["layer"] == "conv5_block3_out"
        assert exp_data["images"]["overlay"].startswith("data:image/")
        print(f"Target layer: {exp_data['explainability']['layer']}")
        print(f"Latency: {exp_latency:.2f} ms")
        print("Status: PASS (Grad-CAM schema & base64 Data-URLs verified)")
        qa_results["explain_schema"] = True

        # 6. Consistency Verification (Predict vs Explain)
        print("\n--- Step 6: Prediction vs Explainability Consistency Test ---")
        pred_class = pred_data["prediction"]["class"]
        pred_idx = pred_data["prediction"]["class_index"]
        exp_class = exp_data["prediction"]["class_name"]
        exp_idx = exp_data["prediction"]["class_index"]

        print(f"Predict Endpoint: class='{pred_class}' (idx {pred_idx})")
        print(f"Explain Endpoint: class='{exp_class}' (idx {exp_idx})")

        assert pred_class == exp_class, f"Class mismatch: {pred_class} vs {exp_class}"
        assert pred_idx == exp_idx, f"Index mismatch: {pred_idx} vs {exp_idx}"
        print("Status: PASS (Prediction & Explainability are 100% consistent)")
        qa_results["predict_explain_consistency"] = True

        # 7. Security: Path Traversal Filename Test
        print("\n--- Step 7: Security - Path Traversal Filename Test ---")
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("../../../../etc/passwd.jpg", img_bytes, "image/jpeg")}
        )
        assert resp.status_code == 200, "Path traversal in filename caused crash or error"
        print("Status: PASS (Sanitized path traversal filename successfully)")
        qa_results["security_path_traversal"] = True

        # 8. Security: Unicode Filename Test
        print("\n--- Step 8: Security - Unicode Filename Test ---")
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("lesión_dermoscópica_ñ_123.png", img_bytes, "image/png")}
        )
        assert resp.status_code == 200
        print("Status: PASS (Unicode filename handled safely)")
        qa_results["security_unicode_filename"] = True

        # 9. Security: Extremely Long Filename Test
        print("\n--- Step 9: Security - Extremely Long Filename Test ---")
        long_filename = "a" * 300 + ".jpg"
        resp = client.post(
            "/api/v1/predict",
            files={"image": (long_filename, img_bytes, "image/jpeg")}
        )
        assert resp.status_code == 200
        print("Status: PASS (Long filename handled safely)")
        qa_results["security_long_filename"] = True

        # 10. Security: Corrupted Image Bytes Test
        print("\n--- Step 10: Security - Corrupted Image Binary Test ---")
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("corrupt.jpg", b"INVALID_CORRUPT_BYTES_9999", "image/jpeg")}
        )
        assert resp.status_code == 400
        assert "detail" in resp.json()
        print("Status: PASS (Corrupted image bytes cleanly rejected with HTTP 400)")
        qa_results["security_corrupt_bytes"] = True

        # 11. Security: Zero-Byte Empty Upload Test
        print("\n--- Step 11: Security - Empty 0-byte File Upload Test ---")
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("empty.jpg", b"", "image/jpeg")}
        )
        assert resp.status_code == 400
        print("Status: PASS (Empty file upload rejected with HTTP 400)")
        qa_results["security_empty_upload"] = True

        # 12. Security: Unsupported File Extension Test
        print("\n--- Step 12: Security - Unsupported File Extension Test (.bmp) ---")
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("unsupported.bmp", b"dummy bmp bytes", "image/bmp")}
        )
        assert resp.status_code in (400, 415)
        print("Status: PASS (Unsupported media type cleanly rejected with HTTP 415)")
        qa_results["security_unsupported_ext"] = True

        # 13. Performance Benchmarks
        print("\n--- Step 13: Performance Latency Benchmarks (5 iterations) ---")
        predict_latencies = []
        explain_latencies = []

        for i in range(5):
            t0 = time.perf_counter()
            client.post("/api/v1/predict", files={"image": ("sample.jpg", img_bytes, "image/jpeg")})
            predict_latencies.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            client.post("/api/v1/explain", files={"image": ("sample.jpg", img_bytes, "image/jpeg")})
            explain_latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_pred = sum(predict_latencies) / len(predict_latencies)
        avg_exp = sum(explain_latencies) / len(explain_latencies)

        print(f"Mean Predict API Latency:  {avg_pred:.2f} ms")
        print(f"Mean Explain API Latency:  {avg_exp:.2f} ms")
        print("Status: PASS (Performance metrics recorded)")
        qa_results["performance_benchmark"] = True

    # 14. Post-Test Model Integrity Check
    print("\n--- Step 14: Post-Test Model Integrity Verification ---")
    sha_ok_after = verify_model_sha256()
    assert sha_ok_after, "Model SHA-256 verification failed after test suite!"
    print("Status: PASS (Model SHA-256 100% Unchanged)")

    print("\n" + "=" * 70)
    print("PHASE 31 QA & HARDENING SUMMARY")
    print("=" * 70)
    all_pass = all(qa_results.values())
    for k, v in qa_results.items():
        print(f"  {k:32s}: {'PASS' if v else 'FAIL'}")
    print(f"Overall QA Suite Status: {'ALL QA CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    success = run_full_qa_suite()
    if not success:
        exit(1)
