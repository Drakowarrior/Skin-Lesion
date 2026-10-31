"""
Lightweight Clinical Image Quality Gate for Skin Lesion Analysis.

Provides fast, explainable pre-inference technical validation of skin lesion
photographs to filter out severely degraded, blurred, or occluded uploads before
passing them to the ResNet50 neural network.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Operates strictly PRE-INFERENCE: answers only technical suitability questions.
2. NEVER makes clinical diagnoses, cancer predictions, or benignity determinations.
3. Lightweight and sub-millisecond execution using standard OpenCV / NumPy routines.
"""

from enum import Enum
import math
from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
from PIL import Image


class QualityState(str, Enum):
    """Tri-tier image quality classification."""
    PASS = "QUALITY_PASS"
    WARNING = "QUALITY_WARNING"
    FAIL = "QUALITY_FAIL"


class ImageQualityGate:
    """
    Evaluates technical suitability of clinical skin lesion photographs.
    
    Attributes:
        min_fail_sharpness (float): Severe blur cutoff (30.0).
        min_warn_sharpness (float): Moderate blur cutoff (70.0).
        min_fail_brightness (float): Severe darkness cutoff (35.0).
        max_fail_brightness (float): Severe glare cutoff (225.0).
        min_warn_brightness (float): Low lighting warning (65.0).
        max_warn_brightness (float): High lighting warning (205.0).
        max_fail_underexposure (float): Severe dark pixel fraction (0.40).
        max_fail_overexposure (float): Severe glare pixel fraction (0.35).
        min_fail_contrast (float): Minimum grayscale std dev (15.0).
        min_fail_entropy (float): Minimum Shannon entropy (4.0).
        min_resolution (int): Minimum image width/height in pixels (112).
    """

    # Validation-Derived Threshold Defaults
    TH_FAIL_SHARPNESS: float = 30.0
    TH_WARN_SHARPNESS: float = 70.0
    
    TH_FAIL_MIN_BRIGHTNESS: float = 35.0
    TH_FAIL_MAX_BRIGHTNESS: float = 225.0
    TH_WARN_MIN_BRIGHTNESS: float = 65.0
    TH_WARN_MAX_BRIGHTNESS: float = 205.0
    
    TH_FAIL_MAX_UNDEREXPOSURE: float = 0.40
    TH_FAIL_MAX_OVEREXPOSURE: float = 0.35
    TH_WARN_MAX_EXPOSURE_DEFECT: float = 0.15
    
    TH_FAIL_MIN_CONTRAST: float = 12.0
    TH_WARN_MIN_CONTRAST: float = 20.0
    
    TH_FAIL_MIN_ENTROPY: float = 4.0
    TH_MIN_RESOLUTION_PX: int = 112

    def __init__(
        self,
        min_fail_sharpness: float = TH_FAIL_SHARPNESS,
        min_warn_sharpness: float = TH_WARN_SHARPNESS,
        min_fail_brightness: float = TH_FAIL_MIN_BRIGHTNESS,
        max_fail_brightness: float = TH_FAIL_MAX_BRIGHTNESS,
        min_warn_brightness: float = TH_WARN_MIN_BRIGHTNESS,
        max_warn_brightness: float = TH_WARN_MAX_BRIGHTNESS,
        max_fail_underexposure: float = TH_FAIL_MAX_UNDEREXPOSURE,
        max_fail_overexposure: float = TH_FAIL_MAX_OVEREXPOSURE,
        min_fail_contrast: float = TH_FAIL_MIN_CONTRAST,
        min_fail_entropy: float = TH_FAIL_MIN_ENTROPY,
        min_resolution: int = TH_MIN_RESOLUTION_PX
    ):
        self.min_fail_sharpness = min_fail_sharpness
        self.min_warn_sharpness = min_warn_sharpness
        self.min_fail_brightness = min_fail_brightness
        self.max_fail_brightness = max_fail_brightness
        self.min_warn_brightness = min_warn_brightness
        self.max_warn_brightness = max_warn_brightness
        self.max_fail_underexposure = max_fail_underexposure
        self.max_fail_overexposure = max_fail_overexposure
        self.min_fail_contrast = min_fail_contrast
        self.min_fail_entropy = min_fail_entropy
        self.min_resolution = min_resolution

    @staticmethod
    def _to_grayscale_numpy(image_input: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert input image (PIL Image or NumPy RGB/BGR/Grayscale array) to uint8 2D grayscale."""
        if isinstance(image_input, Image.Image):
            gray_img = image_input.convert("L")
            return np.array(gray_img, dtype=np.uint8)
        
        arr = np.asarray(image_input)
        if arr.ndim == 2:
            return arr.astype(np.uint8)
        elif arr.ndim == 3:
            if arr.shape[2] == 3:
                # Standard RGB/BGR to Grayscale Rec. 601 luma formula
                gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                return np.clip(gray, 0, 255).astype(np.uint8)
            elif arr.shape[2] == 4:
                gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                return np.clip(gray, 0, 255).astype(np.uint8)
        raise ValueError(f"Unsupported image shape for quality analysis: {arr.shape}")

    @staticmethod
    def _compute_laplacian_variance(gray: np.ndarray) -> float:
        """Compute variance of discrete Laplacian kernel convolution without requiring OpenCV."""
        # 3x3 standard discrete Laplacian kernel
        # [ 0,  1,  0]
        # [ 1, -4,  1]
        # [ 0,  1,  0]
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0
        
        # Fast vectorized discrete Laplacian using central slice
        g_float = gray.astype(np.float64)
        lap = (
            g_float[0:-2, 1:-1] +
            g_float[2:, 1:-1] +
            g_float[1:-1, 0:-2] +
            g_float[1:-1, 2:] -
            4.0 * g_float[1:-1, 1:-1]
        )
        return float(np.var(lap))

    @staticmethod
    def _compute_shannon_entropy(gray: np.ndarray) -> float:
        """Compute Shannon information entropy in bits [0.0, 8.0]."""
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        total_pixels = gray.size
        probs = hist / total_pixels
        # Filter zero probability bins
        non_zero = probs[probs > 0]
        return float(-np.sum(non_zero * np.log2(non_zero)))

    def analyze(self, image_input: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Extract all technical quality signals from the image.
        
        Returns:
            Dictionary containing metrics, state, flags, and user guidance.
        """
        gray = self._to_grayscale_numpy(image_input)
        h, w = gray.shape
        total_pixels = gray.size
        
        # 1. Dimensions
        min_dim = min(h, w)
        aspect_ratio = float(w / h) if h > 0 else 0.0
        
        # 2. Sharpness / Blur
        sharpness = self._compute_laplacian_variance(gray)
        
        # 3. Brightness & Contrast
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # 4. Exposure Defects
        underexposure_fraction = float(np.sum(gray <= 15) / total_pixels)
        overexposure_fraction = float(np.sum(gray >= 240) / total_pixels)
        
        # 5. Shannon Entropy
        entropy = self._compute_shannon_entropy(gray)
        
        # Evaluation flags
        defects = []
        warnings = []
        
        # Check Hard Failures
        if min_dim < self.min_resolution:
            defects.append(f"Image resolution too low ({w}x{h} px; minimum required is {self.min_resolution}x{self.min_resolution} px).")
        if sharpness < self.min_fail_sharpness:
            defects.append(f"Severe image blur detected (sharpness: {sharpness:.1f}; minimum required is {self.min_fail_sharpness:.1f}).")
        if brightness < self.min_fail_brightness:
            defects.append(f"Severe underexposure/darkness detected (mean brightness: {brightness:.1f}/255).")
        if brightness > self.max_fail_brightness:
            defects.append(f"Severe overexposure/glare washout detected (mean brightness: {brightness:.1f}/255).")
        if underexposure_fraction > self.max_fail_underexposure:
            defects.append(f"Severe shadow occlusion ({underexposure_fraction*100:.1f}% pixels underexposed).")
        if overexposure_fraction > self.max_fail_overexposure:
            defects.append(f"Severe specular flash reflection ({overexposure_fraction*100:.1f}% pixels saturated).")
        if contrast < self.min_fail_contrast:
            defects.append(f"Severe low contrast / flat image detected (contrast std: {contrast:.1f}).")
        if entropy < self.min_fail_entropy:
            defects.append(f"Degenerate / blank visual information detected (entropy: {entropy:.2f} bits).")

        # Check Soft Warnings
        if not defects:
            if sharpness < self.min_warn_sharpness:
                warnings.append("Mild out-of-focus blur detected. Image is analyzable but sharpness is suboptimal.")
            if brightness < self.min_warn_brightness:
                warnings.append("Low ambient lighting detected. Lesion features may be partially shaded.")
            if brightness > self.max_warn_brightness:
                warnings.append("Bright direct lighting detected. Minor surface glare present.")
            if underexposure_fraction > self.TH_WARN_MAX_EXPOSURE_DEFECT:
                warnings.append("Moderate peripheral shadowing present.")
            if overexposure_fraction > self.TH_WARN_MAX_EXPOSURE_DEFECT:
                warnings.append("Moderate specular reflection present.")
            if contrast < self.TH_WARN_MIN_CONTRAST:
                warnings.append("Moderate low contrast detected.")

        # Determine State
        if defects:
            state = QualityState.FAIL
            guidance = (
                "The uploaded image is technically unsuitable for accurate skin analysis. "
                f"Issues detected: {' '.join(defects)} Please upload a well-focused, evenly lit photograph."
            )
        elif warnings:
            state = QualityState.WARNING
            guidance = (
                "The image is technically analyzable, but minor quality concerns were noted. "
                f"Notes: {' '.join(warnings)} Analysis will proceed with cautionary differential reporting."
            )
        else:
            state = QualityState.PASS
            guidance = "Image satisfies all clinical quality and technical illumination standards."

        metrics = {
            "width": w,
            "height": h,
            "aspect_ratio": round(aspect_ratio, 2),
            "sharpness_laplacian_var": round(sharpness, 2),
            "brightness_mean": round(brightness, 2),
            "contrast_std": round(contrast, 2),
            "underexposure_fraction": round(underexposure_fraction, 4),
            "overexposure_fraction": round(overexposure_fraction, 4),
            "shannon_entropy_bits": round(entropy, 2)
        }

        return {
            "state": state.value,
            "metrics": metrics,
            "defects": defects,
            "warnings": warnings,
            "guidance_message": guidance,
            "is_acceptable_for_inference": bool(state != QualityState.FAIL)
        }
