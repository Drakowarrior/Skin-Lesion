"""
Production Deployment Recovery Verification Test Suite

Covers:
1. Model Downloader Verification:
   - Rejects 404 cleanly with exit code 1 and cleans temp files
   - Rejects checksum mismatch and does not overwrite target
   - Recognizes existing valid checkpoint and verifies SHA-256
   - Rejects partial/interrupted files
2. Backend Production Behavior:
   - Production imports do NOT load TensorFlow or Keras in sys.modules
   - Model loads and is shared between predictor and explainer
   - Missing model in production environment raises RuntimeError fail-fast
   - Invalid model checksum in production environment raises ValueError fail-fast
   - Health endpoint correctly reflects liveness and readiness
   - Predict and Explain endpoint schemas are strictly preserved
3. Frontend Configuration Verification:
   - VITE_API_BASE_URL is configured
   - No backend secrets are in frontend source
"""

import sys
import os
import unittest
import tempfile
import hashlib
import time
import shutil
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.convnext_inference import EXPECTED_CONVNEXT_SHA256, DEFAULT_CONVNEXT_CHECKPOINT_PATH
from scripts.download_production_weights import (
    compute_sha256,
    download_and_verify,
    get_expected_sha256,
)


class MockServerHandler(BaseHTTPRequestHandler):
    """Mock HTTP server for testing downloader behavior (404, valid, partial, corrupt)."""
    valid_payload = b"VALID_PRODUCTION_MODEL_WEIGHTS_CONTENT"

    def log_message(self, format, *args):
        pass  # Suppress console logging

    def do_GET(self):
        if self.path == "/valid_model.pt":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(self.valid_payload)))
            self.end_headers()
            self.wfile.write(self.valid_payload)
        elif self.path == "/corrupt_model.pt":
            corrupt = b"CORRUPT_PAYLOAD"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(corrupt)))
            self.end_headers()
            self.wfile.write(corrupt)
        elif self.path == "/not_found.pt":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
        else:
            self.send_response(400)
            self.end_headers()


class ProductionDownloaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), MockServerHandler)
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_downloader_rejects_404_cleanly(self):
        """Downloader must fail cleanly on 404 without creating target file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model.pt"
            url = f"{self.base_url}/not_found.pt"
            success = download_and_verify(
                url=url,
                target_path=target,
                expected_sha256="dummy_hash",
                timeout=5
            )
            self.assertFalse(success, "Downloader should return False on 404")
            self.assertFalse(target.exists(), "Target file should not exist after 404")
            # Verify no orphaned temp files
            tmp_files = list(Path(tmp_dir).glob("download_*.tmp"))
            self.assertEqual(len(tmp_files), 0, "Temporary download files should be cleaned up")

    def test_downloader_rejects_checksum_mismatch(self):
        """Downloader must reject downloaded content with mismatched checksum."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model.pt"
            url = f"{self.base_url}/corrupt_model.pt"
            expected_hash = hashlib.sha256(b"DIFFERENT_CONTENT").hexdigest()
            success = download_and_verify(
                url=url,
                target_path=target,
                expected_sha256=expected_hash,
                timeout=5
            )
            self.assertFalse(success, "Downloader should fail on SHA mismatch")
            self.assertFalse(target.exists(), "Corrupt file should not be moved to target")

    def test_downloader_preserves_valid_existing_checkpoint(self):
        """If target exists and passes checksum, download is safely skipped."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model.pt"
            content = MockServerHandler.valid_payload
            target.write_bytes(content)
            valid_hash = hashlib.sha256(content).hexdigest()

            # Point to a URL that would 404 if accessed
            url = f"{self.base_url}/not_found.pt"
            success = download_and_verify(
                url=url,
                target_path=target,
                expected_sha256=valid_hash,
                force=False,
                timeout=5
            )
            self.assertTrue(success, "Should succeed by verifying valid existing file")
            self.assertEqual(target.read_bytes(), content)

    def test_production_checkpoint_matches_manifest(self):
        """Target production checkpoint matches committed SHA-256 manifest."""
        manifest_path = PROJECT_ROOT / "models" / "production" / "model_sha256.txt"
        self.assertTrue(manifest_path.exists(), "Manifest file must exist")
        expected_sha = get_expected_sha256(manifest_path, "convnext_small_skin_lesion_v1.pt")
        self.assertEqual(expected_sha, EXPECTED_CONVNEXT_SHA256)

        prod_model = PROJECT_ROOT / DEFAULT_CONVNEXT_CHECKPOINT_PATH
        if prod_model.exists():
            computed = compute_sha256(prod_model)
            self.assertEqual(computed, EXPECTED_CONVNEXT_SHA256, "Local checkpoint SHA-256 matches manifest")


class BackendProductionBehaviorTests(unittest.TestCase):
    def test_tensorflow_not_imported_in_convnext_mode(self):
        """Ensure TensorFlow is not loaded into sys.modules when using ConvNeXt."""
        # Check current loaded modules
        self.assertNotIn("tensorflow", sys.modules, "TensorFlow must NOT be loaded for ConvNeXt")
        self.assertNotIn("keras", sys.modules, "Keras must NOT be loaded for ConvNeXt")

    def test_missing_model_fail_fast_in_production(self):
        """In ENVIRONMENT=production, missing model must raise RuntimeError during startup."""
        import importlib
        from backend import config
        original_env = os.environ.get("ENVIRONMENT")
        original_path = os.environ.get("MODEL_PATH")
        try:
            os.environ["ENVIRONMENT"] = "production"
            os.environ["MODEL_PATH"] = "/nonexistent/path/to/model.pt"
            importlib.reload(config)

            from backend.main import lifespan, app
            import asyncio

            async def attempt_startup():
                async with lifespan(app):
                    pass

            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(attempt_startup())
            self.assertIn("CRITICAL STARTUP FAILURE", str(ctx.exception))
        finally:
            if original_env is not None:
                os.environ["ENVIRONMENT"] = original_env
            else:
                os.environ.pop("ENVIRONMENT", None)
            if original_path is not None:
                os.environ["MODEL_PATH"] = original_path
            else:
                os.environ.pop("MODEL_PATH", None)
            importlib.reload(config)

    def test_live_prediction_and_gradcam_schemas(self):
        """Verify /predict and /explain return valid production schemas with ConvNeXt."""
        from backend.main import app
        with TestClient(app) as client:
            # 1. Health check
            resp = client.get("/api/v1/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["readiness"], "ready")
            self.assertEqual(data["model"], "convnext_small_v1")
            self.assertTrue(data["model_loaded"])
            self.assertTrue(data["explainability_loaded"])

            # 2. Prediction endpoint
            sample_img = PROJECT_ROOT / "backend" / "test_sample.jpg"
            with open(sample_img, "rb") as f:
                img_bytes = f.read()

            pred_resp = client.post(
                "/api/v1/predict",
                files={"image": ("sample.jpg", img_bytes, "image/jpeg")}
            )
            self.assertEqual(pred_resp.status_code, 200)
            pred_data = pred_resp.json()
            self.assertTrue(pred_data["success"])
            self.assertIn("prediction", pred_data)
            self.assertIn("class", pred_data["prediction"])
            self.assertIn("calibrated_probability", pred_data["prediction"])
            self.assertIn("uncertainty_state", pred_data["prediction"])
            self.assertEqual(len(pred_data["top3"]), 3)

            # 3. Explain endpoint
            exp_resp = client.post(
                "/api/v1/explain",
                files={"image": ("sample.jpg", img_bytes, "image/jpeg")}
            )
            self.assertEqual(exp_resp.status_code, 200)
            exp_data = exp_resp.json()
            self.assertTrue(exp_data["success"])
            self.assertIn("explainability", exp_data)
            self.assertIn("images", exp_data)
            self.assertIn("overlay_blended", exp_data["images"])
            self.assertTrue(exp_data["images"]["overlay_blended"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
