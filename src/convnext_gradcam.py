"""
ConvNeXt-Small Grad-CAM Explainability Module (Phase 12)

Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for the PyTorch ConvNeXt-Small
model using PyTorch forward and backward hooks on the final feature stage.

DISCLAIMER: Grad-CAM visualizations highlight image regions that contributed
statistically to the model's prediction. They do NOT represent a confirmed medical
diagnosis, clinical proof of malignancy, or diagnostic certainty.
"""

import io
import base64
import logging
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn.functional as F

from src.class_labels import CLASS_NAMES
from src.convnext_inference import ConvNeXtLesionPredictor

logger = logging.getLogger("src.convnext_gradcam")

SAFETY_EXPLAINABILITY_DISCLAIMER = (
    "Grad-CAM visualization shows image regions that contributed relatively more to the model prediction. "
    "It is an interpretability aid, not a pixel-level medical segmentation or diagnostic certainty."
)


class ConvNeXtGradCAMGenerator:
    """
    Production-grade Grad-CAM generator for PyTorch ConvNeXt-Small.
    Hooks into `model.features[-1]` to generate pixel-aligned, transparent colormap overlays.
    """
    def __init__(self, predictor: ConvNeXtLesionPredictor):
        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device

        # Target the final feature block before pooling: features[7] or features[-1]
        self.target_layer = self.model.features[-1]
        self.target_layer_name = "features.7 (ConvNeXtBlock Stage 4)"

        self.activations = None
        self.gradients = None
        self._register_hooks()
        logger.info(f"ConvNeXt Grad-CAM initialized | Target layer: '{self.target_layer_name}'")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def compute_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class_index: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Grad-CAM heatmap array normalized [0.0, 1.0] for target_class_index.
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        logits = self.model(image_tensor) # Shape: (1, 14)
        score = logits[0, target_class_index]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        # Gradients: (1, C, H, W) -> Global Average Pooling across spatial dims
        # In ConvNeXt features[-1], output shape is (1, 768, 12, 12)
        grads = self.gradients.detach()
        acts = self.activations.detach()

        # Immediate memory cleanup of hook buffers and parameter gradients
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        # Channel weights: alpha_k = mean(grad_k)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam) # Apply ReLU

        cam_np = cam.squeeze().cpu().numpy() # (H, W)

        # Normalize heatmap to [0.0, 1.0]
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)
        if max_val - min_val > 1e-8:
            heatmap_norm = (cam_np - min_val) / (max_val - min_val)
        else:
            heatmap_norm = np.zeros_like(cam_np)

        # Quantitative Metrics
        peak_act = float(np.max(heatmap_norm))
        focus_ratio = float(np.mean(heatmap_norm > 0.5))

        metrics = {
            "target_layer": self.target_layer_name,
            "target_class": CLASS_NAMES[target_class_index],
            "target_class_index": int(target_class_index),
            "peak_activation": peak_act,
            "focus_ratio": focus_ratio,
            "interpretation": (
                "Highly focused focal activation on salient lesion pathology."
                if focus_ratio < 0.25 else
                "Diffuse activation across multiple lesion structural patterns."
            )
        }

        return heatmap_norm, metrics

    def generate_explanation(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        target_class_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate complete Grad-CAM explainability response with overlay images
        rendered onto original input image dimensions.
        """
        # Load and sanitize PIL Image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input if image_input.dtype == np.uint8 else (image_input * 255).astype(np.uint8))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        orig_w, orig_h = pil_img.size

        # Preprocess for model
        tensor = self.predictor.transform(pil_img).unsqueeze(0).to(self.device)

        # Determine target class if not specified
        if target_class_index is None:
            with torch.no_grad():
                pred_logits = self.model(tensor)
                target_class_index = int(torch.argmax(pred_logits, dim=-1).item())

        # Compute heatmap
        heatmap_lowres, metrics = self.compute_heatmap(tensor, target_class_index)

        # Upsample heatmap to original image resolution with bilinear interpolation
        heatmap_img = Image.fromarray((heatmap_lowres * 255).astype(np.uint8), mode="L")
        heatmap_upsampled = heatmap_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        heatmap_arr = np.array(heatmap_upsampled, dtype=np.float32) / 255.0

        # Apply Jet colormap
        colormap = matplotlib.colormaps.get_cmap("jet")
        colored_heatmap_arr = colormap(heatmap_arr)[:, :, :3] # Shape: (H, W, 3) in [0, 1]
        colored_heatmap_uint8 = (colored_heatmap_arr * 255).astype(np.uint8)
        colored_heatmap_pil = Image.fromarray(colored_heatmap_uint8, mode="RGB")

        # Create Intensity-Modulated Alpha Blend Overlay
        orig_arr = np.array(pil_img, dtype=np.float32) / 255.0
        # Alpha modulated by activation intensity: max 0.65 alpha in high-activation zones
        alpha = np.expand_dims(heatmap_arr * 0.60, axis=-1)
        blended_arr = (orig_arr * (1.0 - alpha)) + (colored_heatmap_arr * alpha)
        blended_uint8 = np.clip(blended_arr * 255.0, 0, 255).astype(np.uint8)
        blended_pil = Image.fromarray(blended_uint8, mode="RGB")

        # 7x7 spatial activation grid
        grid_7x7 = [[round(float(v), 4) for v in row] for row in heatmap_lowres]
        metrics["heatmap_grid"] = grid_7x7

        # Encode images to base64 Data URLs
        def to_data_url(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"

        orig_url = to_data_url(pil_img)
        heatmap_url = to_data_url(colored_heatmap_pil)
        overlay_url = to_data_url(blended_pil)

        return {
            "success": True,
            "prediction": {
                "class_name": CLASS_NAMES[target_class_index],
                "class_index": int(target_class_index),
            },
            "explainability": metrics,
            "heatmap": grid_7x7,
            "images": {
                "original_preview": orig_url,
                "heatmap_colored": heatmap_url,
                "overlay_blended": overlay_url,
                "original": orig_url,
                "heatmap": heatmap_url,
                "overlay": overlay_url,
            },
            "disclaimer": SAFETY_EXPLAINABILITY_DISCLAIMER
        }
