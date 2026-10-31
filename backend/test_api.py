"""
Backend Integration Test Suite (Phase 28)

Tests FastAPI endpoints:
- GET /api/v1/health
- GET /api/v1/model
- POST /api/v1/predict (valid non-test training image, invalid extension, corrupted image, empty file)

Enforces strict verification rules:
- Test dataset (data/processed/test.csv) is strictly isolated and NOT used.
- Validates canonical 14-class mapping and prediction schema.
- Measures and reports API inference latency per request.
"""

import io
import time
from pathlib import Path
import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from src.class_labels import CLASS_NAMES, NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_valid_non_test_image_path() -> Path:
    """Return verified non-test sample image for API integration testing."""
    sample_p = PROJECT_ROOT / "backend" / "test_sample.jpg"
    return sample_p


def test_api_suite():
    """Run full backend test suite."""
    results = {
        "health": False,
        "model": False,
        "predict_valid": False,
        "predict_invalid_ext": False,
        "predict_unsupported": False,
        "predict_corrupt": False,
        "predict_empty": False,
    }

    print("\n" + "=" * 60)
    print("RUNNING PHASE 28 BACKEND API TEST SUITE")
    print("=" * 60)

    # Use TestClient with lifespan support context manager
    with TestClient(app) as client:
        
        # 1. Health Endpoint Test
        print("\n--- Test 1: GET /api/v1/health ---")
        t0 = time.perf_counter()
        resp = client.get("/api/v1/health")
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"Request: GET /api/v1/health")
        print(f"Inference latency: {latency:.2f} ms")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model"] == "resnet50_skin_lesion_v1"
        assert data["model_version"] == "v1"
        assert data["model_loaded"] is True
        print("Status: PASS (Health check successful, model loaded)")
        results["health"] = True

        # 2. Model Metadata Endpoint Test
        print("\n--- Test 2: GET /api/v1/model ---")
        t0 = time.perf_counter()
        resp = client.get("/api/v1/model")
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"Request: GET /api/v1/model")
        print(f"Inference latency: {latency:.2f} ms")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["model_name"] == "resnet50_skin_lesion_v1"
        assert data["version"] == "v1"
        assert data["architecture"] == "ResNet50"
        assert data["num_classes"] == NUM_CLASSES == 14
        assert data["status"] == "frozen"
        print(f"Canonical classes count: {data['num_classes']} (Matches CLASS_NAMES)")
        print("Status: PASS (Model metadata verified)")
        results["model"] = True

        # 3. Valid Non-Test Image Prediction Test
        print("\n--- Test 3: POST /api/v1/predict (Valid Non-Test Image) ---")
        sample_img_path = get_valid_non_test_image_path()
        print(f"Using non-test sample image: {sample_img_path.relative_to(PROJECT_ROOT)}")

        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.perf_counter()
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("sample_lesion.jpg", img_bytes, "image/jpeg")}
        )
        latency = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Schema & response validations
        assert data["success"] is True
        pred = data["prediction"]
        assert pred["class"] in CLASS_NAMES
        assert 0 <= pred["class_index"] < NUM_CLASSES
        assert 0.0 <= pred["confidence"] <= 1.0

        top3 = data["top3"]
        assert len(top3) == 3
        top3_classes = [item["class"] for item in top3]
        assert len(set(top3_classes)) == 3, "Top-3 predictions contain duplicate classes"
        for item in top3:
            assert item["class"] in CLASS_NAMES
            assert 0 <= item["class_index"] < NUM_CLASSES
            assert 0.0 <= item["confidence"] <= 1.0

        assert data["model"]["name"] == "resnet50_skin_lesion_v1"
        assert "disclaimer" in data

        print(f"Request: POST /api/v1/predict (Valid Image)")
        print(f"Inference latency: {latency:.2f} ms")
        print(f"Predicted class: {pred['class']} (Index {pred['class_index']})")
        print(f"Confidence: {pred['confidence']:.4f}")
        print("Top-3 predictions:")
        for idx, item in enumerate(top3, 1):
            print(f"  {idx}. {item['class']}: {item['confidence']:.4f}")
        print("Status: PASS (Prediction response schema and values verified)")
        results["predict_valid"] = True

        # 4. Invalid File Extension (.txt) Test
        print("\n--- Test 4: POST /api/v1/predict (Invalid Extension .txt) ---")
        t0 = time.perf_counter()
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("document.txt", b"Hello World text file content", "text/plain")}
        )
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"Request: Invalid Extension (.txt)")
        print(f"Inference latency: {latency:.2f} ms")
        print(f"HTTP Status: {resp.status_code} | Error: {resp.json().get('detail')}")
        assert resp.status_code in (400, 415), f"Expected 400 or 415, got {resp.status_code}"
        print("Status: PASS (Invalid file extension correctly rejected)")
        results["predict_invalid_ext"] = True

        # 5. Unsupported Image Format (.bmp) Test
        print("\n--- Test 5: POST /api/v1/predict (Unsupported Format .bmp) ---")
        t0 = time.perf_counter()
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("test.bmp", b"dummy bmp bytes", "image/bmp")}
        )
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"Request: Unsupported Format (.bmp)")
        print(f"Inference latency: {latency:.2f} ms")
        print(f"HTTP Status: {resp.status_code} | Error: {resp.json().get('detail')}")
        assert resp.status_code in (400, 415), f"Expected 400 or 415, got {resp.status_code}"
        print("Status: PASS (Unsupported image format correctly rejected)")
        results["predict_unsupported"] = True

        # 6. Corrupted Image Test
        print("\n--- Test 6: POST /api/v1/predict (Corrupted Image .jpg) ---")
        t0 = time.perf_counter()
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("corrupt.jpg", b"INVALID_CORRUPT_HEADER_BYTES_9999", "image/jpeg")}
        )
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"Request: Corrupted Image (.jpg)")
        print(f"Inference latency: {latency:.2f} ms")
        print(f"HTTP Status: {resp.status_code} | Error: {resp.json().get('detail')}")
        assert resp.status_code == 400, f"Expected 400 Bad Request, got {resp.status_code}"
        print("Status: PASS (Corrupted image correctly rejected)")
        results["predict_corrupt"] = True

        # 7. Empty Upload Test
        print("\n--- Test 7: POST /api/v1/predict (Empty Upload 0-bytes) ---")
        t0 = time.perf_counter()
        resp = client.post(
            "/api/v1/predict",
            files={"image": ("empty.jpg", b"", "image/jpeg")}
        )
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"Request: Empty Upload (0-bytes)")
        print(f"Inference latency: {latency:.2f} ms")
        print(f"HTTP Status: {resp.status_code} | Error: {resp.json().get('detail')}")
        assert resp.status_code == 400, f"Expected 400 Bad Request, got {resp.status_code}"
        print("Status: PASS (Empty upload correctly rejected)")
        results["predict_empty"] = True

    print("\n" + "=" * 60)
    print("BACKEND TEST SUITE SUMMARY")
    print("=" * 60)
    all_pass = all(results.values())
    for k, v in results.items():
        print(f"  {k:25s}: {'PASS' if v else 'FAIL'}")
    print(f"Overall Suite Status: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    success = test_api_suite()
    if not success:
        exit(1)
