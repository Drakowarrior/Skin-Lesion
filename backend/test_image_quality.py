"""
Unit tests for the ImageQualityGate module (src/image_quality.py).
"""

import unittest
import numpy as np
from PIL import Image
from src.image_quality import ImageQualityGate, QualityState


class TestImageQualityGate(unittest.TestCase):
    def setUp(self):
        self.gate = ImageQualityGate()

    def test_sharp_good_image_passes(self):
        # Create a synthetic skin lesion-like image with rich continuous tone gradient and texture
        np.random.seed(42)
        base = np.random.normal(loc=140.0, scale=15.0, size=(224, 224, 3))
        # Add gradient illumination
        y, x = np.ogrid[:224, :224]
        gradient = (x / 224.0) * 20.0 + (y / 224.0) * 15.0
        base += gradient[:, :, np.newaxis]
        # Add high-contrast textured circular lesion
        mask = (x - 112)**2 + (y - 112)**2 <= 50**2
        base[mask] -= 50.0
        # Add high frequency lesion microstructure
        texture = np.random.normal(loc=0.0, scale=12.0, size=(224, 224, 3))
        base[mask] += texture[mask]
        arr = np.clip(base, 0, 255).astype(np.uint8)
        
        res = self.gate.analyze(arr)
        self.assertEqual(res["state"], QualityState.PASS.value)
        self.assertTrue(res["is_acceptable_for_inference"])
        self.assertEqual(len(res["defects"]), 0)
        self.assertGreater(res["metrics"]["sharpness_laplacian_var"], 70.0)
        self.assertGreater(res["metrics"]["shannon_entropy_bits"], 4.5)

    def test_blurry_uniform_image_fails(self):
        # Create a completely flat / blurry image (zero sharpness)
        arr = np.full((224, 224, 3), 140, dtype=np.uint8)
        res = self.gate.analyze(arr)
        self.assertEqual(res["state"], QualityState.FAIL.value)
        self.assertFalse(res["is_acceptable_for_inference"])
        self.assertTrue(any("blur" in d.lower() or "contrast" in d.lower() or "entropy" in d.lower() for d in res["defects"]))

    def test_pitch_black_image_fails(self):
        # Completely dark image
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        res = self.gate.analyze(arr)
        self.assertEqual(res["state"], QualityState.FAIL.value)
        self.assertFalse(res["is_acceptable_for_inference"])
        self.assertTrue(any("dark" in d.lower() or "underexposure" in d.lower() for d in res["defects"]))

    def test_specular_whiteout_image_fails(self):
        # Completely white saturated image
        arr = np.full((224, 224, 3), 250, dtype=np.uint8)
        res = self.gate.analyze(arr)
        self.assertEqual(res["state"], QualityState.FAIL.value)
        self.assertFalse(res["is_acceptable_for_inference"])
        self.assertTrue(any("glare" in d.lower() or "overexposure" in d.lower() for d in res["defects"]))

    def test_low_resolution_fails(self):
        # Tiny 50x50 image
        arr = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)
        res = self.gate.analyze(arr)
        self.assertEqual(res["state"], QualityState.FAIL.value)
        self.assertTrue(any("resolution" in d.lower() for d in res["defects"]))

    def test_pil_image_support(self):
        # Ensure PIL image objects can be analyzed directly
        pil_img = Image.new("RGB", (224, 224), color=(140, 130, 120))
        res = self.gate.analyze(pil_img)
        self.assertIn("metrics", res)
        self.assertEqual(res["metrics"]["width"], 224)
        self.assertEqual(res["metrics"]["height"], 224)


if __name__ == "__main__":
    unittest.main()
