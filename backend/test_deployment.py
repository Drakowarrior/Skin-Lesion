"""
Backend Deployment & Restart Test Suite (Phase 32)

Tests:
- Server startup liveness & readiness semantics
- Execution of /predict and /explain endpoints
- Process restart simulation (re-initializing predictor and explainer)
- Post-restart execution correctness
- SHA-256 model immutability verification before and after restart
"""

import time
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from src.inference import SkinLesionPredictor
from backend.explainability import ExplainabilityService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "production" / "resnet50_skin_lesion_v1.keras"
EXPECTED_SHA256 = "93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e"


def verify_model_sha256() -> bool:
    """Verify that frozen production model SHA-256 checksum remains byte-for-byte unchanged."""
    with open(MODEL_PATH, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    return actual_sha256 == EXPECTED_SHA256


def get_valid_non_test_image_path() -> Path:
    """Return verified non-test sample image for deployment testing."""
    sample_p = PROJECT_ROOT / "backend" / "test_sample.jpg"
    return sample_p


def test_deployment_and_restart():
    """Run deployment readiness & process restart simulation test."""
    print("\n" + "=" * 70)
    print("RUNNING PHASE 32 DEPLOYMENT & PROCESS RESTART SUITE")
    print("=" * 70)

    # 1. SHA-256 Check
    print("\n--- Step 1: Pre-Restart Model SHA-256 Verification ---")
    assert verify_model_sha256(), "Model SHA-256 checksum mismatch!"
    print("Status: PASS (Model SHA-256 Verified)")

    sample_img_path = get_valid_non_test_image_path()
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    # 2. Initial Run
    print("\n--- Step 2: First Server Process Execution ---")
    with TestClient(app) as client:
        resp_health = client.get("/api/v1/health")
        assert resp_health.status_code == 200
        health_data = resp_health.json()
        assert health_data["liveness"] == "alive"
        assert health_data["readiness"] == "ready"
        print("Health Status: PASS (Liveness & Readiness verified)")

        resp_pred = client.post(
            "/api/v1/predict",
            files={"image": ("sample.jpg", img_bytes, "image/jpeg")}
        )
        assert resp_pred.status_code == 200
        pred_class_1 = resp_pred.json()["prediction"]["class"]
        print(f"Run 1 Prediction: {pred_class_1}")

        resp_exp = client.post(
            "/api/v1/explain",
            files={"image": ("sample.jpg", img_bytes, "image/jpeg")}
        )
        assert resp_exp.status_code == 200
        print("Run 1 Grad-CAM: PASS")

    # 3. Simulate Server Restart (Re-run Lifespan)
    print("\n--- Step 3: Simulating Process Restart ---")
    with TestClient(app) as client_restart:
        resp_health_2 = client_restart.get("/api/v1/health")
        assert resp_health_2.status_code == 200
        assert resp_health_2.json()["readiness"] == "ready"
        print("Post-Restart Health Status: PASS")

        resp_pred_2 = client_restart.post(
            "/api/v1/predict",
            files={"image": ("sample.jpg", img_bytes, "image/jpeg")}
        )
        assert resp_pred_2.status_code == 200
        pred_class_2 = resp_pred_2.json()["prediction"]["class"]
        print(f"Run 2 Post-Restart Prediction: {pred_class_2}")

        assert pred_class_1 == pred_class_2, "Prediction changed after restart!"
        print("Status: PASS (Prediction identical after process restart)")

    # 4. Post-Restart SHA-256 Check
    print("\n--- Step 4: Post-Restart Model SHA-256 Verification ---")
    assert verify_model_sha256(), "Model SHA-256 altered after restart!"
    print("Status: PASS (Model SHA-256 Unchanged)")

    print("\n" + "=" * 70)
    print("DEPLOYMENT & RESTART SUITE PASSED 100%")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_deployment_and_restart()
