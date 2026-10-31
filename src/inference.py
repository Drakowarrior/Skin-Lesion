"""
Production Skin Lesion Inference Module (Phase 27)

Provides a robust, reusable inference API for the frozen ResNet50 skin lesion classifier.
Handles image loading, channel normalization, format validation, and prediction formatting.

DISCLAIMER: Output probabilities represent statistical model confidence scores for
research and demonstration purposes only, NOT diagnostic medical certainty or safety guarantees.
"""

from pathlib import Path
from typing import Dict, Any, Union, List
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from src.class_labels import CLASS_NAMES, INDEX_TO_CLASS, NUM_CLASSES
from src.calibration import TemperatureScaler
from src.uncertainty import UncertaintyEstimator, PredictionState
from src.image_quality import ImageQualityGate, QualityState


DEFAULT_PRODUCTION_MODEL_PATH = Path("models/production/resnet50_skin_lesion_v1.keras")
FALLBACK_MODEL_PATH = Path("models/resnet50_phase26_best.keras")


class SkinLesionPredictor:
    """
    Production Predictor for ResNet50 Skin Lesion Classifier with integrated
    post-hoc Temperature Scaling and Uncertainty Quantification.
    """
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        temperature: float = 1.3421,
        enable_calibration: bool = True
    ):
        if model_path is None:
            if DEFAULT_PRODUCTION_MODEL_PATH.exists():
                model_path = DEFAULT_PRODUCTION_MODEL_PATH
            elif FALLBACK_MODEL_PATH.exists():
                model_path = FALLBACK_MODEL_PATH
            else:
                raise FileNotFoundError(
                    f"No model checkpoint found at '{DEFAULT_PRODUCTION_MODEL_PATH}' or '{FALLBACK_MODEL_PATH}'."
                )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load frozen model in read-only mode
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.input_shape = self.model.input_shape[1:3]  # (224, 224)
        self.num_classes = self.model.output_shape[-1]  # 14

        if self.num_classes != NUM_CLASSES:
            raise ValueError(
                f"Model output dimension ({self.num_classes}) does not match expected class count ({NUM_CLASSES})."
            )

        # Post-hoc Calibration, Uncertainty, and Quality Gate Engines
        self.enable_calibration = enable_calibration
        self.scaler = TemperatureScaler(temperature=temperature)
        self.uncertainty_estimator = UncertaintyEstimator(scaler=self.scaler, num_classes=self.num_classes)
        self.quality_gate = ImageQualityGate()

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Validate, normalize, and preprocess image input into ResNet50 input format.

        Handles:
        - Image file paths (str or Path)
        - PIL Image objects (RGB, RGBA, Grayscale 'L', Palette 'P')
        - NumPy arrays

        Returns:
            Preprocessed image batch tensor of shape (1, 224, 224, 3)
        """
        # 1. Load PIL Image
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
                # Ensure uint8 array format
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

        # 2. Format Normalization (Convert Grayscale / RGBA / Palette to 3-Channel RGB)
        try:
            pil_img = ImageOps.exif_transpose(pil_img)  # Respect EXIF rotation tags
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Error converting image to RGB format: {str(e)}")

        # 3. Resize to Model Input Shape (224, 224)
        target_w, target_h = self.input_shape
        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # 4. Convert to Float32 NumPy Array (224, 224, 3)
        img_arr = np.array(pil_img, dtype=np.float32)

        # 5. Apply ResNet50 Preprocessing (Caffe-style BGR channel flip + ImageNet mean subtraction)
        img_prep = tf.keras.applications.resnet50.preprocess_input(img_arr)

        # 6. Expand batch dimension -> (1, 224, 224, 3)
        batch = np.expand_dims(img_prep, axis=0)
        return batch

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Run inference on image input and return structured prediction results
        including calibrated probabilities and uncertainty decision state.

        Returns:
            Dictionary containing predicted class, index, calibrated confidence,
            uncertainty state, probabilities dict, and top-3 predictions.
        """
        # Quality assessment if image input is available as PIL/array
        quality_res = None
        if isinstance(image_input, (Image.Image, np.ndarray, str, Path)):
            try:
                if isinstance(image_input, (str, Path)):
                    p_img = Image.open(image_input)
                elif isinstance(image_input, np.ndarray):
                    p_img = Image.fromarray(image_input if image_input.dtype == np.uint8 else (image_input * 255).astype(np.uint8))
                else:
                    p_img = image_input
                quality_res = self.quality_gate.analyze(p_img)
            except Exception:
                quality_res = None

        batch = self.preprocess_image(image_input)
        raw_probs = self.model.predict(batch, verbose=0)[0]  # Shape (14,)

        # Validate probabilities array
        if not np.all(np.isfinite(raw_probs)):
            raise ValueError("Inference output contains NaN or Inf values.")

        # Apply post-hoc temperature scaling calibration
        if self.enable_calibration:
            calibrated_probs = self.scaler.calibrate_probabilities(raw_probs)
        else:
            calibrated_probs = raw_probs.copy()

        # Evaluate uncertainty metrics and decision state
        unc_metrics = self.uncertainty_estimator.compute_metrics(calibrated_probs)
        state, clinical_msg, diag_flags = self.uncertainty_estimator.evaluate_state(unc_metrics)
        evidence = self.uncertainty_estimator.compute_evidence_strength(unc_metrics, state)

        best_idx = int(np.argmax(calibrated_probs))
        cal_confidence = float(calibrated_probs[best_idx])
        raw_confidence = float(raw_probs[best_idx])

        # Top-3 predictions (sorted by calibrated probabilities)
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

