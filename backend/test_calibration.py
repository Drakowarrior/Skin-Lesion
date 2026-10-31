"""
Unit tests for the TemperatureScaler calibration module (src/calibration.py).
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
from src.calibration import TemperatureScaler


class TestTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.scaler = TemperatureScaler(temperature=1.3421)

    def test_initialization(self):
        self.assertEqual(self.scaler.temperature, 1.3421)
        with self.assertRaises(ValueError):
            TemperatureScaler(temperature=-1.0)
        with self.assertRaises(ValueError):
            TemperatureScaler(temperature=0.0)

    def test_temperature_setter(self):
        self.scaler.temperature = 1.5
        self.assertEqual(self.scaler.temperature, 1.5)
        with self.assertRaises(ValueError):
            self.scaler.temperature = -0.5

    def test_calibrate_logits_sum_to_one(self):
        logits = np.array([2.5, 0.1, -1.2, 5.8, 1.0])
        cal_probs = self.scaler.calibrate_logits(logits)
        self.assertAlmostEqual(float(np.sum(cal_probs)), 1.0, places=6)
        self.assertTrue(np.all(cal_probs >= 0.0))
        self.assertTrue(np.all(cal_probs <= 1.0))

    def test_calibrate_logits_preserves_ranking(self):
        logits = np.array([1.2, 3.4, 0.5, 8.9, -2.1, 4.0])
        cal_probs = self.scaler.calibrate_logits(logits)
        self.assertEqual(np.argmax(logits), np.argmax(cal_probs))
        self.assertTrue(np.array_equal(np.argsort(logits), np.argsort(cal_probs)))

    def test_calibrate_probabilities_softening(self):
        # Raw saturated probabilities
        raw_probs = np.array([0.9999, 0.00005, 0.00005])
        cal_probs = self.scaler.calibrate_probabilities(raw_probs)
        self.assertAlmostEqual(float(np.sum(cal_probs)), 1.0, places=6)
        # Saturated top prob should soften under T > 1
        self.assertLess(cal_probs[0], raw_probs[0])
        self.assertEqual(np.argmax(raw_probs), np.argmax(cal_probs))

    def test_batch_calibration(self):
        logits_batch = np.array([
            [10.0, 2.0, 1.0],
            [1.0, 8.0, 3.0],
            [0.5, 0.5, 4.0]
        ])
        cal_probs = self.scaler.calibrate_logits(logits_batch)
        self.assertEqual(cal_probs.shape, (3, 3))
        np.testing.assert_allclose(np.sum(cal_probs, axis=1), [1.0, 1.0, 1.0], atol=1e-6)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "calib.json"
            self.scaler.save(config_path)
            self.assertTrue(config_path.exists())
            
            loaded_scaler = TemperatureScaler.load(config_path)
            self.assertAlmostEqual(loaded_scaler.temperature, 1.3421, places=4)


if __name__ == "__main__":
    unittest.main()
