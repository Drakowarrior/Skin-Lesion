"""
Unit tests for the UncertaintyEstimator and Abstention module (src/uncertainty.py).
"""

import unittest
import numpy as np
from src.uncertainty import UncertaintyEstimator, PredictionState
from src.calibration import TemperatureScaler


class TestUncertaintyEstimator(unittest.TestCase):
    def setUp(self):
        self.scaler = TemperatureScaler(temperature=1.3421)
        self.estimator = UncertaintyEstimator(scaler=self.scaler, num_classes=14)

    def test_compute_metrics_shapes(self):
        # 14-class probability distribution
        probs = np.zeros(14)
        probs[0] = 0.80
        probs[1] = 0.10
        probs[2] = 0.05
        probs[3:] = 0.05 / 11
        
        metrics = self.estimator.compute_metrics(probs)
        self.assertAlmostEqual(metrics["top1_probability"], 0.80, places=4)
        self.assertAlmostEqual(metrics["top2_probability"], 0.10, places=4)
        self.assertAlmostEqual(metrics["top3_probability"], 0.05, places=4)
        self.assertAlmostEqual(metrics["prediction_margin"], 0.70, places=4)
        self.assertGreater(metrics["predictive_entropy"], 0.0)
        self.assertLessEqual(metrics["normalized_entropy"], 1.0)
        self.assertGreaterEqual(metrics["normalized_entropy"], 0.0)

    def test_state_evaluation_confident(self):
        # High confidence, wide margin, low entropy
        probs = np.zeros(14)
        probs[9] = 0.90
        probs[10] = 0.05
        probs[0] = 0.05
        
        metrics = self.estimator.compute_metrics(probs)
        state, msg, flags = self.estimator.evaluate_state(metrics)
        self.assertEqual(state, PredictionState.CONFIDENT)
        self.assertFalse(flags["low_confidence"])
        self.assertFalse(flags["narrow_margin"])
        self.assertFalse(flags["high_entropy"])

    def test_state_evaluation_uncertain(self):
        # Moderate confidence (0.55), moderate margin (0.20), moderate entropy
        probs = np.zeros(14)
        probs[9] = 0.55
        probs[10] = 0.35
        probs[1] = 0.10
        
        metrics = self.estimator.compute_metrics(probs)
        state, msg, flags = self.estimator.evaluate_state(metrics)
        self.assertEqual(state, PredictionState.UNCERTAIN)
        self.assertTrue(flags["borderline_confidence"] or flags["borderline_margin"] or flags["borderline_entropy"])

    def test_state_evaluation_abstain_flat_distribution(self):
        # Uniform flat distribution (e.g. noise or OOD-like degenerate input)
        probs = np.ones(14) / 14.0
        
        metrics = self.estimator.compute_metrics(probs)
        state, msg, flags = self.estimator.evaluate_state(metrics)
        self.assertEqual(state, PredictionState.ABSTAIN)
        self.assertTrue(flags["low_confidence"])
        self.assertTrue(flags["narrow_margin"])
        self.assertTrue(flags["high_entropy"])
        self.assertAlmostEqual(metrics["normalized_entropy"], 1.0, places=4)

    def test_process_logits_end_to_end(self):
        # Test logits forward processing
        logits = np.array([12.0, 1.0, 0.5, 0.2, 0.1, 0.0, -1.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.1, -0.5])
        res = self.estimator.process_logits(logits)
        
        self.assertIn("calibrated_probabilities", res)
        self.assertIn("metrics", res)
        self.assertIn("state", res)
        self.assertEqual(res["state"], PredictionState.CONFIDENT.value)
        self.assertEqual(len(res["calibrated_probabilities"]), 14)
        self.assertAlmostEqual(sum(res["calibrated_probabilities"]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
