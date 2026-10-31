"""
Backend Explainability Service Module (Phase 14 Production)

Wraps Grad-CAM generators for API request processing.
Supports both PyTorch ConvNeXt-Small and Keras ResNet50 predictors seamlessly.
Uses lazy imports to prevent importing TensorFlow/Keras unless ResNet50 rollback is active.
"""

from typing import Dict, Any, Union, TYPE_CHECKING
from PIL import Image

from src.convnext_gradcam import ConvNeXtGradCAMGenerator
from src.convnext_inference import ConvNeXtLesionPredictor

if TYPE_CHECKING:
    from src.inference import SkinLesionPredictor


class ExplainabilityService:
    """Service wrapper for generating model explainability artifacts."""
    def __init__(self, predictor: Union[ConvNeXtLesionPredictor, "SkinLesionPredictor"]):
        if isinstance(predictor, ConvNeXtLesionPredictor):
            self.generator = ConvNeXtGradCAMGenerator(predictor)
        else:
            # Lazy import TensorFlow-based GradCAM only if using legacy Keras ResNet50 predictor
            from src.gradcam import GradCAMGenerator
            self.generator = GradCAMGenerator(predictor)

    def generate_explanation(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Generate Grad-CAM heatmap and overlay base64 data URLs for image input."""
        return self.generator.generate_explanation(pil_image)

    def generate_explainability(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Alias for generate_explanation."""
        return self.generate_explanation(pil_image)
