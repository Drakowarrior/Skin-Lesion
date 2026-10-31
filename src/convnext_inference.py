"""
ConvNeXt-Small Isolated Production-Ready Inference Adapter (Phase 12)

Provides an isolated, high-performance inference engine for the verified ConvNeXt-Small
challenger model (PyTorch 2.3+ with Apple Silicon MPS acceleration).

DISCLAIMER: Output probabilities represent statistical model confidence scores for
research and demonstration purposes only, NOT diagnostic medical certainty or safety guarantees.
"""

from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import hashlib
import time
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from src.class_labels import CLASS_NAMES, INDEX_TO_CLASS, NUM_CLASSES
from src.calibration import TemperatureScaler
from src.uncertainty import UncertaintyEstimator, PredictionState
from src.image_quality import ImageQualityGate, QualityState

DEFAULT_CONVNEXT_CHECKPOINT_PATH = Path("models/production/convnext_small_skin_lesion_v1.pt")
EXPECTED_CONVNEXT_SHA256 = "67f4d8d04edef9389aeac77688dcc90e193b717de67f8c4c0c1ebf15483a3fe3"


class ConvNeXtLesionPredictor:
    """
    Isolated Predictor for ConvNeXt-Small Skin Lesion Classifier with integrated
    Image Quality Gate, Temperature Scaling, and Uncertainty Quantification.
    """
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        temperature: float = 1.0000,
        enable_calibration: bool = True,
        device: Optional[torch.device] = None,
        verify_sha: bool = True
    ):
        if model_path is None:
            model_path = DEFAULT_CONVNEXT_CHECKPOINT_PATH
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ConvNeXt checkpoint file not found at '{self.model_path}'. "
                "Explicit fail-fast: silent fallback to ResNet50 is prohibited."
            )

        # 1. Checkpoint Checksum Verification
        if verify_sha:
            with open(self.model_path, "rb") as f:
                actual_sha = hashlib.sha256(f.read()).hexdigest()
            if actual_sha != EXPECTED_CONVNEXT_SHA256:
                raise ValueError(
                    f"ConvNeXt checkpoint SHA-256 mismatch! "
                    f"Expected: {EXPECTED_CONVNEXT_SHA256}, Got: {actual_sha}"
                )

        # 2. Compute Device Configuration
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.input_shape = (384, 384)
        self.num_classes = NUM_CLASSES

        # 3. Model Architecture Construction & State Loading
        self.model = models.convnext_small(weights=None)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, self.num_classes)
        )

        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # 4. Preprocessing Pipeline Definition
        self.transform = transforms.Compose([
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 5. Quality Gate, Calibration, and Uncertainty Engines
        self.enable_calibration = enable_calibration
        self.scaler = TemperatureScaler(temperature=temperature)
        self.uncertainty_estimator = UncertaintyEstimator(scaler=self.scaler, num_classes=self.num_classes)
        self.quality_gate = ImageQualityGate()

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Validate, normalize, and preprocess image input into PyTorch ConvNeXt input tensor.
        Shape: (1, 3, 384, 384) on compute device.
        """
        pil_img = None
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            try:
                pil_img = Image.open(path)
            except Exception as e:
                raise ValueError(f"Failed to open image file '{path}': {str(e)}")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, np.ndarray):
            try:
                if image_input.dtype != np.uint8:
                    if image_input.max() <= 1.0:
                        image_input = (image_input * 255.0).astype(np.uint8)
                    else:
                        image_input = image_input.astype(np.uint8)
                pil_img = Image.fromarray(image_input)
            except Exception as e:
                raise ValueError(f"Failed to convert NumPy array to PIL Image: {str(e)}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Format Normalization
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Error converting image to RGB format: {str(e)}")

        # Apply transforms & add batch dimension
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Run inference on image input and return structured prediction results
        including calibrated probabilities, uncertainty state, and top-3 differential classes.
        """
        # 1. Pre-inference Quality Assessment
        quality_res = None
        p_img = None
        if isinstance(image_input, (str, Path)):
            p_img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            p_img = image_input
        elif isinstance(image_input, np.ndarray):
            p_img = Image.fromarray(image_input if image_input.dtype == np.uint8 else (image_input * 255).astype(np.uint8))

        if p_img is not None:
            quality_res = self.quality_gate.analyze(p_img)

        # 2. Preprocess and Tensor Forward Pass
        tensor = self.preprocess_image(image_input)
        with torch.no_grad():
            logits = self.model(tensor) # Shape: (1, 14)
            raw_probs_tensor = F.softmax(logits, dim=-1)
            raw_probs = raw_probs_tensor.cpu().numpy()[0] # Shape: (14,)

        # 3. Floating-Point Validation
        if not np.all(np.isfinite(raw_probs)):
            raise ValueError("Inference output contains NaN or Inf values.")

        # 4. Calibration (T = 1.0000)
        if self.enable_calibration:
            calibrated_probs = self.scaler.calibrate_probabilities(raw_probs)
        else:
            calibrated_probs = raw_probs.copy()

        # 5. Uncertainty Quantification & Decision State
        unc_metrics = self.uncertainty_estimator.compute_metrics(calibrated_probs)
        state, clinical_msg, diag_flags = self.uncertainty_estimator.evaluate_state(unc_metrics)
        evidence = self.uncertainty_estimator.compute_evidence_strength(unc_metrics, state)

        best_idx = int(np.argmax(calibrated_probs))
        cal_confidence = float(calibrated_probs[best_idx])
        raw_confidence = float(raw_probs[best_idx])

        # 6. Top-3 Differential Predictions
        top3_indices = np.argsort(calibrated_probs)[::-1][:3]
        top3_list = [
            {
                "class": CLASS_NAMES[idx],
                "index": int(idx),
                "confidence": float(calibrated_probs[idx]),
                "raw_confidence": float(raw_probs[idx]),
            }
            for idx in top3_indices
        ]

        cal_prob_dict = {
            CLASS_NAMES[i]: float(calibrated_probs[i])
            for i in range(NUM_CLASSES)
        }
        raw_prob_dict = {
            CLASS_NAMES[i]: float(raw_probs[i])
            for i in range(NUM_CLASSES)
        }

        return {
            "predicted_class": CLASS_NAMES[best_idx],
            "predicted_index": best_idx,
            "confidence": cal_confidence,
            "raw_confidence": raw_confidence,
            "temperature_used": self.scaler.temperature,
            "is_calibrated": self.enable_calibration,
            "probability_sum": float(np.sum(calibrated_probs)),
            "probabilities": cal_prob_dict,
            "raw_probabilities": raw_prob_dict,
            "top3": top3_list,
            "evidence_strength": evidence,
            "uncertainty": {
                "state": state.value,
                "metrics": unc_metrics,
                "clinical_message": clinical_msg,
                "diagnostic_flags": diag_flags,
            },
            "quality": quality_res
        }
