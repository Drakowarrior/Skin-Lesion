"""
Unit tests for Grad-CAM explainability module validation (src/gradcam.py).
"""

import unittest
import numpy as np
from PIL import Image
from src.inference import SkinLesionPredictor
from src.gradcam import GradCAMGenerator


class TestGradCAMValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SkinLesionPredictor()
        cls.gradcam = GradCAMGenerator(cls.predictor)
        cls.test_pil = Image.open("backend/test_sample.jpg")

    def test_target_layer_and_shape(self):
        self.assertEqual(self.gradcam.target_layer_name, "conv5_block3_out")
        batch = self.predictor.preprocess_image(self.test_pil)
        raw_heatmap, metrics = self.gradcam.compute_heatmap(batch, target_class_index=7)
        self.assertEqual(raw_heatmap.shape, (7, 7))
        self.assertEqual(metrics["feature_map_shape"], [1, 7, 7, 2048])
        self.assertGreaterEqual(float(np.min(raw_heatmap)), 0.0)
        self.assertLessEqual(float(np.max(raw_heatmap)), 1.0)

    def test_determinism(self):
        res1 = self.gradcam.generate_explanation(self.test_pil, target_class_index=7)
        res2 = self.gradcam.generate_explanation(self.test_pil, target_class_index=7)
        grid1 = np.array(res1["explainability"]["heatmap_grid"])
        grid2 = np.array(res2["explainability"]["heatmap_grid"])
        np.testing.assert_allclose(grid1, grid2, atol=1e-6)

    def test_class_target_discrimination(self):
        # Grad-CAM for class 7 vs class 11 vs class 3
        res1 = self.gradcam.generate_explanation(self.test_pil, target_class_index=7)
        res2 = self.gradcam.generate_explanation(self.test_pil, target_class_index=11)
        grid1 = np.array(res1["explainability"]["heatmap_grid"])
        grid2 = np.array(res2["explainability"]["heatmap_grid"])
        # Should not be identical
        diff = float(np.mean(np.abs(grid1 - grid2)))
        self.assertGreater(diff, 0.01)

    def test_coordinate_query_and_out_of_bounds(self):
        res = self.gradcam.generate_explanation(self.test_pil, target_class_index=7)
        grid = res["explainability"]["heatmap_grid"]
        w = res["explainability"]["heatmap_width"]
        h = res["explainability"]["heatmap_height"]
        
        # Valid center query
        val_center = self.gradcam.get_activation_at_coordinate(grid, w/2, h/2, w, h)
        self.assertEqual(val_center["status"], "available")
        self.assertEqual(val_center["grid_row"], 3)
        self.assertEqual(val_center["grid_col"], 3)
        self.assertIsNotNone(val_center["activation"])
        
        # Out-of-bounds query
        oob_neg = self.gradcam.get_activation_at_coordinate(grid, -10, 50, w, h)
        self.assertEqual(oob_neg["status"], "unavailable")
        self.assertIsNone(oob_neg["activation"])
        
        oob_pos = self.gradcam.get_activation_at_coordinate(grid, w + 100, h/2, w, h)
        self.assertEqual(oob_pos["status"], "unavailable")
        self.assertIsNone(oob_pos["activation"])

    def test_peak_activation_properties(self):
        res = self.gradcam.generate_explanation(self.test_pil, target_class_index=7)
        peak = res["explainability"]["peak_activation"]
        self.assertAlmostEqual(peak["value"], 1.0, places=4)
        self.assertGreaterEqual(peak["grid_row"], 0)
        self.assertLess(peak["grid_row"], 7)
        self.assertGreaterEqual(peak["grid_col"], 0)
        self.assertLess(peak["grid_col"], 7)
        self.assertGreater(peak["pixel_x"], 0)
        self.assertGreater(peak["pixel_y"], 0)


if __name__ == "__main__":
    unittest.main()
