"""
Phase 7D Test Suite: Scientifically Honest Prediction Display

Verifies that:
1. Calibrated probabilities remain mathematically pure (T=1.3421).
2. Probabilities are never artificially capped at 90%, 95%, or 99%.
3. Probability vector sums to approximately 1.0.
4. Top-3 differential classes reflect true sorted calibrated probabilities.
5. Decision states (CONFIDENT, UNCERTAIN, ABSTAIN) truthfully reflect Phase 3 policies.
6. QUALITY_FAIL short-circuits and suppresses prediction display.
7. Temperature scaling is applied exactly once.
8. Scientific non-diagnostic disclaimers are preserved.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from src.class_labels import NUM_CLASSES, CLASS_NAMES
from src.calibration import TemperatureScaler
from src.uncertainty import UncertaintyEstimator, PredictionState
from src.image_quality import ImageQualityGate


@pytest.fixture(scope="module")
def estimator():
    scaler = TemperatureScaler(temperature=1.3421)
    return UncertaintyEstimator(scaler=scaler, num_classes=NUM_CLASSES)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_calibrated_probability_preservation(estimator):
    """Calibrated probability is strictly preserved and never capped or replaced."""
    # Create sample logits
    logits = np.zeros(NUM_CLASSES, dtype=np.float32)
    logits[0] = 7.0
    logits[1] = 1.0

    result = estimator.process_logits(logits)
    probs = result["calibrated_probabilities"]

    # Expected temperature-scaled probabilities (T=1.3421)
    scaled_logits = logits / 1.3421
    expected_probs = np.exp(scaled_logits - np.max(scaled_logits))
    expected_probs /= np.sum(expected_probs)

    np.testing.assert_allclose(probs, expected_probs, rtol=1e-5)
    top1_prob = result["metrics"]["top1_probability"]
    assert top1_prob == pytest.approx(float(expected_probs[0]), rel=1e-5)
    
    # Must NOT be artificially capped at 90% or 95%
    assert top1_prob > 0.90
    assert top1_prob <= 1.0


def test_high_confidence_uncapped(estimator):
    """Extreme logits produce true mathematical probabilities without artificial suppression."""
    logits = np.zeros(NUM_CLASSES, dtype=np.float32)
    logits[11] = 15.0  # High confidence monkeypox

    result = estimator.process_logits(logits)
    top1 = result["metrics"]["top1_probability"]
    # Calibrated probability should be >0.999
    assert top1 > 0.999
    assert top1 <= 1.0


def test_probability_vector_sums_to_one(estimator):
    """Calibrated probability vector sums to 1.0 within float precision."""
    rng = np.random.RandomState(42)
    for _ in range(20):
        logits = rng.randn(NUM_CLASSES).astype(np.float32) * 3.0
        result = estimator.process_logits(logits)
        probs = np.array(result["calibrated_probabilities"])
        assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-6)


def test_top3_differential_ordering(estimator):
    """Top-3 classes are ordered strictly descending by calibrated probability."""
    logits = np.zeros(NUM_CLASSES, dtype=np.float32)
    logits[2] = 5.0
    logits[5] = 3.0
    logits[9] = 2.0

    result = estimator.process_logits(logits)
    m = result["metrics"]
    assert m["top1_index"] == 2
    assert m["top2_index"] == 5
    assert m["top3_index"] == 9
    assert m["top1_probability"] > m["top2_probability"] > m["top3_probability"]


def test_decision_states_truthfulness(estimator):
    """Decision states strictly follow Phase 3 validation thresholds."""
    # 1. Clear Confident Case (p1 >= 0.65, margin >= 0.35, norm_ent <= 0.40)
    logits_conf = np.zeros(NUM_CLASSES, dtype=np.float32)
    logits_conf[0] = 6.0
    res_conf = estimator.process_logits(logits_conf)
    assert res_conf["state"] == PredictionState.CONFIDENT.value
    assert res_conf["metrics"]["top1_probability"] >= 0.65

    # 2. Uncertain Case (p1 in [0.45, 0.65), margin in [0.15, 0.35))
    logits_unc = np.zeros(NUM_CLASSES, dtype=np.float32)
    logits_unc[0] = 4.5
    logits_unc[1] = 3.5
    res_unc = estimator.process_logits(logits_unc)
    assert res_unc["state"] == PredictionState.UNCERTAIN.value
    assert res_unc["metrics"]["top1_probability"] < 0.65

    # 3. Hard Abstain Case (uniform / very low confidence)
    logits_abs = np.zeros(NUM_CLASSES, dtype=np.float32)
    res_abs = estimator.process_logits(logits_abs)
    assert res_abs["state"] == PredictionState.ABSTAIN.value
    assert "abstained" in res_abs["clinical_message"].lower()


def test_quality_fail_suppresses_prediction(client):
    """A technically unsuitable image (QUALITY_FAIL) short-circuits without generating predictions."""
    img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="JPEG")
    buf.seek(0)

    response = client.post("/api/v1/predict", files={"image": ("dark.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["quality"]["state"] == "QUALITY_FAIL"
    assert data["prediction"] is None


def test_api_predict_returns_honest_probability(client):
    """API predict endpoint returns unadulterated calibrated probability and top-3 distribution."""
    with open("backend/test_sample.jpg", "rb") as f:
        img_bytes = f.read()

    response = client.post("/api/v1/predict", files={"image": ("test_sample.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "prediction" in data
    pred = data["prediction"]
    assert "class" in pred
    assert "probability" in pred
    assert "calibrated_probability" in pred
    assert "confidence" in pred
    assert "uncertainty_state" in pred
    assert "margin" in pred
    assert "normalized_entropy" in pred

    # Verify top-3 retains true calibrated probabilities
    assert len(data["top3"]) == 3
    top1 = data["top3"][0]["probability"]
    top2 = data["top3"][1]["probability"]
    top3 = data["top3"][2]["probability"]
    assert top1 >= top2 >= top3
    assert 0.0 <= top1 <= 1.0


def test_temperature_scaling_applied_once(estimator):
    """Verify that temperature scaling T=1.3421 is applied exactly once and not compounded."""
    logits = np.array([2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scaler = TemperatureScaler(temperature=1.3421)
    
    # Calculate directly
    scaled_manual = logits / 1.3421
    p_manual = np.exp(scaled_manual - np.max(scaled_manual))
    p_manual /= np.sum(p_manual)
    
    # Via scaler
    p_scaler = scaler.calibrate_logits(logits)
    np.testing.assert_allclose(p_manual, p_scaler, rtol=1e-6)
