"""
Production Grad-CAM Explainability Module (Clean Single-Overlay Architecture)

Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for the frozen ResNet50 classifier
without modifying model weights.

DISCLAIMER: Grad-CAM visualizations highlight image regions that contributed
statistically to the model's prediction. They do NOT represent a confirmed medical
diagnosis, clinical proof of malignancy, or diagnostic certainty.
"""

import io
import math
import base64
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.cm as cm
import tensorflow as tf

from src.class_labels import CLASS_NAMES
from src.inference import SkinLesionPredictor

logger = logging.getLogger("src.gradcam")

SAFETY_EXPLAINABILITY_DISCLAIMER = (
    "Grad-CAM visualization shows image regions that contributed relatively more to the model prediction. "
    "It is an interpretability aid, not a pixel-level medical segmentation or diagnostic certainty."
)


class GradCAMGenerator:
    """
    Production Grad-CAM generator for frozen ResNet50 model.
    Generates pixel-aligned, transparent colormap overlays on original input image dimensions.
    """
    def __init__(self, predictor: SkinLesionPredictor):
        self.predictor = predictor
        self.model = predictor.model

        # Programmatically inspect model architecture to locate target convolutional layer
        self.resnet_submodel = self.model.get_layer("resnet50")
        self.target_layer_name = "conv5_block3_out"

        try:
            self.target_layer = self.resnet_submodel.get_layer(self.target_layer_name)
        except ValueError:
            # Search for final conv layer before global pooling
            conv_layers = [
                l for l in self.resnet_submodel.layers
                if "conv5" in l.name or "out" in l.name or isinstance(l, tf.keras.layers.Conv2D)
            ]
            if not conv_layers:
                raise ValueError("Could not find any suitable target convolutional layer in model.")
            self.target_layer = conv_layers[-1]
            self.target_layer_name = self.target_layer.name

        logger.info(f"Grad-CAM initialized | Target layer: '{self.target_layer_name}'")

        # Submodel for gradient computation: input -> (target_conv_out, resnet_out)
        self.sub_grad_model = tf.keras.models.Model(
            inputs=self.resnet_submodel.inputs,
            outputs=[self.target_layer.output, self.resnet_submodel.output]
        )

        # Classification head layers
        self.top_layers = [
            self.model.get_layer(name)
            for name in ["global_avg_pool", "head_batch_norm", "dense_256", "dropout_0.3", "softmax_output"]
        ]

    def _forward_pass(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Perform forward pass through backbone and top classification layers."""
        conv_out, resnet_out = self.sub_grad_model(x, training=False)
        h = resnet_out
        for layer in self.top_layers:
            if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization)):
                h = layer(h, training=False)
            else:
                h = layer(h)
        return conv_out, h

    def compute_heatmap(
        self,
        image_batch: np.ndarray,
        target_class_index: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Grad-CAM heatmap array normalized [0.0, 1.0] for target_class_index.
        """
        input_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self._forward_pass(input_tensor)
            score = predictions[:, target_class_index]

        grads = tape.gradient(score, conv_outputs)
        if grads is None:
            raise ValueError("Gradients could not be calculated for target class.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        # Apply ReLU to retain positive contributions only
        heatmap = np.maximum(heatmap, 0.0)

        raw_min = float(np.min(heatmap))
        raw_max = float(np.max(heatmap))
        raw_mean = float(np.mean(heatmap))

        # Safe normalization to [0.0, 1.0]
        if raw_max > raw_min and raw_max > 1e-8:
            heatmap = (heatmap - raw_min) / (raw_max - raw_min)
        else:
            heatmap = np.zeros_like(heatmap)

        metrics = {
            "target_layer": self.target_layer_name,
            "feature_map_shape": list(conv_outputs.shape),
            "gradient_shape": list(grads.shape),
            "min_activation": round(float(np.min(heatmap)), 4),
            "max_activation": round(float(np.max(heatmap)), 4),
            "mean_activation": round(float(np.mean(heatmap)), 4),
            "raw_min": round(raw_min, 4),
            "raw_max": round(raw_max, 4),
            "raw_mean": round(raw_mean, 4),
            "method": "Grad-CAM",
        }

        return heatmap, metrics

    @staticmethod
    def _pil_to_base64_data_url(pil_img: Image.Image, format_str: str = "PNG") -> str:
        """Convert PIL Image to base64 data-URL string."""
        buf = io.BytesIO()
        pil_img.save(buf, format=format_str)
        b64_bytes = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{format_str.lower()};base64,{b64_bytes}"

    def generate_explanation(
        self,
        pil_image: Image.Image,
        target_class_index: Optional[int] = None,
        base_alpha: float = 0.40,
        colormap_name: str = "jet"
    ) -> Dict[str, Any]:
        """
        Generate single transparent Grad-CAM overlay image matched to original image dimensions.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        orig_w, orig_h = pil_image.size

        # 1. Run inference to get target class index and probability
        pred_res = self.predictor.predict(pil_image)
        if target_class_index is None:
            target_class_index = pred_res["predicted_index"]

        target_class_name = CLASS_NAMES[target_class_index]
        confidence = float(pred_res["probabilities"][target_class_name])

        # 2. Preprocess image using production pipeline
        batch = self.predictor.preprocess_image(pil_image)

        # 3. Compute spatial Grad-CAM matrix
        raw_heatmap, metrics = self.compute_heatmap(batch, target_class_index)

        # 4. Resize heatmap to exact original image dimensions (W, H)
        heatmap_pil = Image.fromarray((raw_heatmap * 255.0).astype(np.uint8))
        heatmap_resized_pil = heatmap_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        heatmap_norm = np.array(heatmap_resized_pil, dtype=np.float32) / 255.0

        # Renormalize resized matrix to ensure full [0, 1] range
        h_max = float(np.max(heatmap_norm))
        h_min = float(np.min(heatmap_norm))
        if h_max > h_min:
            heatmap_norm = (heatmap_norm - h_min) / (h_max - h_min)

        # 5. Apply Colormap (Jet: Blue -> Cyan -> Green -> Yellow -> Red)
        try:
            colormap = matplotlib.colormaps[colormap_name]
        except AttributeError:
            colormap = cm.get_cmap(colormap_name)

        colored_heatmap_rgb = colormap(heatmap_norm)[:, :, :3]
        colored_heatmap_uint8 = (colored_heatmap_rgb * 255.0).astype(np.uint8)
        colored_heatmap_pil = Image.fromarray(colored_heatmap_uint8)

        # 6. Create Alpha-Blended Single Overlay
        orig_np = np.array(pil_image, dtype=np.float32)
        # Intensity-modulated alpha blending: low Grad-CAM areas stay clear original image, high areas show attention
        effective_alpha = base_alpha * heatmap_norm[:, :, np.newaxis]
        overlay_np = (orig_np * (1.0 - effective_alpha) + colored_heatmap_uint8.astype(np.float32) * effective_alpha).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        # 7. Convert to Base64 Data URLs
        orig_data_url = self._pil_to_base64_data_url(pil_image, "JPEG")
        heatmap_data_url = self._pil_to_base64_data_url(colored_heatmap_pil, "PNG")
        overlay_data_url = self._pil_to_base64_data_url(overlay_pil, "PNG")

        # Grid representation for backend metrics / tests
        grid_7x7 = [[round(float(v), 4) for v in row] for row in raw_heatmap]

        # Extract peak activation coordinates
        peak_y_idx, peak_x_idx = np.unravel_index(np.argmax(raw_heatmap), raw_heatmap.shape)
        peak_x_px = int(round((peak_x_idx + 0.5) * (orig_w / 7.0)))
        peak_y_px = int(round((peak_y_idx + 0.5) * (orig_h / 7.0)))

        logger.info(
            f"Grad-CAM generated | Target Class: '{target_class_name}' ({confidence:.4f}) | "
            f"Layer: '{self.target_layer_name}' | Original Dim: {orig_w}x{orig_h} | Overlay Dim: {orig_w}x{orig_h} | "
            f"Peak: ({peak_x_px}, {peak_y_px})"
        )

        return {
            "success": True,
            "prediction": {
                "class_name": target_class_name,
                "class_index": target_class_index,
                "confidence": round(confidence, 4),
            },
            "explainability": {
                "target_class": target_class_name,
                "target_class_index": target_class_index,
                "target_class_score": round(confidence, 4),
                "target_layer": self.target_layer_name,
                "layer": self.target_layer_name,
                "feature_map_shape": metrics["feature_map_shape"],
                "heatmap_width": orig_w,
                "heatmap_height": orig_h,
                "grid_width": 7,
                "grid_height": 7,
                "heatmap_grid": grid_7x7,
                "peak_activation": {
                    "value": round(float(np.max(raw_heatmap)), 4),
                    "grid_row": int(peak_y_idx),
                    "grid_col": int(peak_x_idx),
                    "pixel_x": peak_x_px,
                    "pixel_y": peak_y_px,
                    "description": "Highest model activation region."
                },
                "min_activation": metrics["min_activation"],
                "max_activation": metrics["max_activation"],
                "mean_activation": metrics["mean_activation"],
                "method": metrics["method"],
            },
            "heatmap": grid_7x7,
            "images": {
                "overlay": overlay_data_url,
                "original": orig_data_url,
                "heatmap": heatmap_data_url,
            },
            "disclaimer": SAFETY_EXPLAINABILITY_DISCLAIMER,
        }

    @staticmethod
    def get_activation_at_coordinate(
        heatmap_grid: list,
        x: float,
        y: float,
        img_w: int,
        img_h: int
    ) -> Dict[str, Any]:
        """
        Query activation level at pixel coordinates (x, y).
        Safely returns 'Activation unavailable' for out-of-bounds coordinates.
        """
        if x < 0 or x >= img_w or y < 0 or y >= img_h or img_w <= 0 or img_h <= 0:
            return {
                "status": "unavailable",
                "message": "Activation unavailable (coordinate outside image boundaries)",
                "activation": None
            }
        
        col = int(min(max(math.floor((x / img_w) * 7), 0), 6))
        row = int(min(max(math.floor((y / img_h) * 7), 0), 6))
        val = float(heatmap_grid[row][col])
        
        return {
            "status": "available",
            "grid_row": row,
            "grid_col": col,
            "activation": round(val, 4),
            "activation_percentage": f"{round(val * 100, 1)}%",
            "semantic_level": "Highest Activation" if val > 0.75 else ("Moderate Activation" if val > 0.40 else "Lower Activation")
        }

