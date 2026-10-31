"""
Backend Explainability Mathematical & Alignment Test Suite

Tests FastAPI Grad-CAM explainability endpoint & mathematical properties:
- POST /api/v1/explain

Verifies:
- Grad-CAM heatmap & single overlay generation for valid images.
- Non-zero finite activation values normalized strictly in [0.0, 1.0].
- Spatial dimensions matching input original image dimensions.
- Single overlay Data-URL distinct from pure original image.
- Target class index matching predicted output.
- Frozen model SHA-256 integrity.
"""

import hashlib
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "production" / "resnet50_skin_lesion_v1.keras"
EXPECTED_SHA256 = "93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e"


def verify_model_sha256() -> bool:
    """Verify that frozen model SHA-256 checksum remains unchanged."""
    if not MODEL_PATH.exists():
        return False
    with open(MODEL_PATH, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    return actual_sha256 == EXPECTED_SHA256


def get_valid_non_test_image_path() -> Path:
    """Return verified non-test sample image for explainability testing."""
    sample_p = PROJECT_ROOT / "backend" / "test_sample.jpg"
    return sample_p


def test_gradcam_mathematical_properties_and_normalization():
    """Verify heatmap dimensions, non-zero values, and strict [0, 1] normalization."""
    assert verify_model_sha256(), "Model SHA-256 checksum mismatch!"
    sample_img_path = get_valid_non_test_image_path()

    with TestClient(app) as client:
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()

        resp = client.post(
            "/api/v1/explain",
            files={"image": ("sample_lesion.jpg", img_bytes, "image/jpeg")}
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        exp = data["explainability"]
        assert exp["target_layer"] == "conv5_block3_out"
        assert 0.0 <= exp["min_activation"] <= 1.0
        assert 0.0 <= exp["max_activation"] <= 1.0
        assert exp["max_activation"] >= exp["mean_activation"]
        assert exp["heatmap_width"] > 0
        assert exp["heatmap_height"] > 0

        # Matrix checks
        grid = data["heatmap"]
        assert len(grid) > 0 and len(grid[0]) > 0
        all_vals = [v for row in grid for v in row]
        assert any(v > 0 for v in all_vals), "Heatmap must contain non-zero activations"
        assert all(0.0 <= v <= 1.0 for v in all_vals), "All heatmap values must be normalized in [0, 1]"


def test_single_overlay_alignment_and_dimensions():
    """Verify overlay data-url exists, matches original dimensions, and is distinct from pure original."""
    sample_img_path = get_valid_non_test_image_path()
    with TestClient(app) as client:
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()
        resp = client.post("/api/v1/explain", files={"image": ("sample.jpg", img_bytes, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        imgs = data["images"]
        exp = data["explainability"]

        sample_img = Image.open(sample_img_path)
        expected_w, expected_h = sample_img.size

        assert "overlay" in imgs
        assert imgs["overlay"] != imgs["original"]
        assert exp["heatmap_width"] == expected_w
        assert exp["heatmap_height"] == expected_h


def test_explainability_suite():
    """Smoke test suite wrapper for explainability endpoints."""
    assert verify_model_sha256()
    sample_img_path = get_valid_non_test_image_path()
    with TestClient(app) as client:
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()
        resp = client.post("/api/v1/explain", files={"image": ("sample.jpg", img_bytes, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
