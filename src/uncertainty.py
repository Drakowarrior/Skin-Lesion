"""
Uncertainty Estimation & Abstention Engine for Skin Lesion Classification.

Provides mathematical quantification of predictive uncertainty and post-hoc
abstention state evaluation based on validation-derived thresholds.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Operates strictly post-hoc on top of calibrated probabilities.
2. Never modifies, fine-tunes, or mutates the frozen production model weights.
3. Preserves clinical safety disclaimers and provides tri-state categorization.
"""

from enum import Enum
import math
from typing import Dict, Any, List, Union, Optional, Tuple
import numpy as np
from src.calibration import TemperatureScaler


class PredictionState(str, Enum):
    """Tri-state categorization for model inference reliability."""
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    ABSTAIN = "ABSTAIN"


class UncertaintyEstimator:
    """
    Evaluates predictive uncertainty metrics and determines abstention states
    using empirical validation-derived thresholds.
    
    Attributes:
        num_classes (int): Total number of target classes (14 for skin lesions).
        scaler (TemperatureScaler): Calibrator instance for scaling logits.
        min_confident_prob (float): Threshold for confident top-1 probability (0.65).
        min_confident_margin (float): Threshold for confident margin (0.35).
        max_confident_entropy (float): Threshold for confident normalized entropy (0.40).
        min_acceptable_prob (float): Hard floor for top-1 probability (0.45).
        min_acceptable_margin (float): Hard floor for margin (0.15).
        max_acceptable_entropy (float): Hard ceiling for normalized entropy (0.60).
    """

    DEFAULT_NUM_CLASSES: int = 14
    
    # Validation-derived acceptance thresholds
    TH_CONFIDENT_PROB: float = 0.65
    TH_CONFIDENT_MARGIN: float = 0.35
    TH_CONFIDENT_ENTROPY: float = 0.40
    
    # Hard abstention thresholds
    TH_ABSTAIN_PROB: float = 0.45
    TH_ABSTAIN_MARGIN: float = 0.15
    TH_ABSTAIN_ENTROPY: float = 0.60

    def __init__(
        self,
        scaler: Optional[TemperatureScaler] = None,
        num_classes: int = DEFAULT_NUM_CLASSES,
        min_confident_prob: float = TH_CONFIDENT_PROB,
        min_confident_margin: float = TH_CONFIDENT_MARGIN,
        max_confident_entropy: float = TH_CONFIDENT_ENTROPY,
        min_acceptable_prob: float = TH_ABSTAIN_PROB,
        min_acceptable_margin: float = TH_ABSTAIN_MARGIN,
        max_acceptable_entropy: float = TH_ABSTAIN_ENTROPY
    ):
        self.scaler = scaler or TemperatureScaler()
        self.num_classes = num_classes
        self.log_num_classes = math.log(num_classes)
        
        self.min_confident_prob = min_confident_prob
        self.min_confident_margin = min_confident_margin
        self.max_confident_entropy = max_confident_entropy
        
        self.min_acceptable_prob = min_acceptable_prob
        self.min_acceptable_margin = min_acceptable_margin
        self.max_acceptable_entropy = max_acceptable_entropy

    def compute_metrics(
        self,
        probabilities: Union[np.ndarray, List[float]],
        eps: float = 1e-12
    ) -> Dict[str, Any]:
        """
        Compute uncertainty and entropy metrics from a calibrated probability vector.
        
        Args:
            probabilities: 1D probability distribution summing to 1.0.
            eps: Numerical stability floor for log computation.
            
        Returns:
            Dictionary containing top-k confidences, margin, entropy, and normalized entropy.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 1 or len(probs) != self.num_classes:
            raise ValueError(f"Expected 1D array of shape ({self.num_classes},), got shape {probs.shape}")
        
        # Sort probabilities descending
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        top1_prob = float(sorted_probs[0])
        top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        top3_prob = float(sorted_probs[2]) if len(sorted_probs) > 2 else 0.0
        
        # Prediction Margin
        margin = float(top1_prob - top2_prob)
        
        # Predictive Entropy: H(p) = -sum(p * log(p))
        p_clipped = np.clip(probs, eps, 1.0)
        entropy = float(-np.sum(p_clipped * np.log(p_clipped)))
        
        # Normalized Entropy: H_norm in [0, 1]
        normalized_entropy = float(entropy / self.log_num_classes)
        
        return {
            "top1_probability": top1_prob,
            "top2_probability": top2_prob,
            "top3_probability": top3_prob,
            "prediction_margin": margin,
            "predictive_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "top1_index": int(sorted_indices[0]),
            "top2_index": int(sorted_indices[1]) if len(sorted_indices) > 1 else None,
            "top3_index": int(sorted_indices[2]) if len(sorted_indices) > 2 else None,
        }

    def evaluate_state(self, metrics: Dict[str, Any]) -> Tuple[PredictionState, str, Dict[str, Any]]:
        """
        Determine decision state and clinical safety messaging.
        
        Returns:
            Tuple of (PredictionState, clinical_message, diagnostic_flags).
        """
        top1 = metrics["top1_probability"]
        margin = metrics["prediction_margin"]
        norm_ent = metrics["normalized_entropy"]
        
        flags = {
            "low_confidence": bool(top1 < self.min_acceptable_prob),
            "narrow_margin": bool(margin < self.min_acceptable_margin),
            "high_entropy": bool(norm_ent > self.max_acceptable_entropy),
            "borderline_confidence": bool(self.min_acceptable_prob <= top1 < self.min_confident_prob),
            "borderline_margin": bool(self.min_acceptable_margin <= margin < self.min_confident_margin),
            "borderline_entropy": bool(self.max_confident_entropy < norm_ent <= self.max_acceptable_entropy)
        }
        
        # Hard Abstention Rule (State 3)
        if flags["low_confidence"] or flags["narrow_margin"] or flags["high_entropy"]:
            state = PredictionState.ABSTAIN
            message = (
                "This image does not provide sufficient discriminatory evidence for a reliable "
                "classification. The prediction has been abstained to prevent potential diagnostic "
                "error. Please consult a qualified dermatologist for clinical evaluation."
            )
            return state, message, flags
            
        # Confident Acceptance Rule (State 1)
        if (top1 >= self.min_confident_prob and 
            margin >= self.min_confident_margin and 
            norm_ent <= self.max_confident_entropy):
            state = PredictionState.CONFIDENT
            message = (
                "Strong discriminatory evidence identified. The classification satisfies rigorous "
                "validation-derived reliability criteria."
            )
            return state, message, flags
            
        # Ambiguous / Uncertain Zone (State 2)
        state = PredictionState.UNCERTAIN
        message = (
            "Ambiguous or borderline lesion presentation. The model identified multiple plausible "
            "differential categories with moderate uncertainty. Professional clinical examination is strongly advised."
        )
        return state, message, flags

    def compute_evidence_strength(
        self,
        metrics: Dict[str, Any],
        state: Union[PredictionState, str]
    ) -> Dict[str, Any]:
        """
        Computes a non-probabilistic presentation metric ('Evidence Strength')
        derived from calibrated probabilities, margin, entropy, and uncertainty state.

        CRITICAL SCIENTIFIC PRINCIPLES:
        1. This is NOT a probability and does NOT replace calibrated probabilities.
        2. Closed-form, monotonic, and 100% deterministic (no stochastic noise).
        3. For CONFIDENT state: scores in range [80, 90] -> 'Strong evidence' or 'Very strong evidence'.
        4. For UNCERTAIN state: scores in range [70, 79] -> 'Limited evidence' or 'Moderate evidence'.
        5. For ABSTAIN state: score is None, label is 'Insufficient evidence'.
        """
        state_str = state.value if isinstance(state, PredictionState) else str(state)

        if state_str == PredictionState.ABSTAIN.value:
            return {
                "score": None,
                "label": "Insufficient evidence",
                "display": "N/A",
                "is_probabilistic": False,
                "disclaimer": "Evidence strength is a presentation score derived from the calibrated model output and uncertainty measures. It is not a medical probability or diagnosis."
            }

        top1 = float(metrics["top1_probability"])
        margin = float(metrics["prediction_margin"])
        norm_ent = float(metrics["normalized_entropy"])

        # Composite discriminatory index c in [0, 1]
        c = 0.50 * top1 + 0.30 * margin + 0.20 * (1.0 - norm_ent)

        if state_str == PredictionState.CONFIDENT.value:
            # c ranges approximately from 0.55 (boundary) to 1.00 (ideal)
            # Map linearly into [80, 90]
            c_norm = float(np.clip((c - 0.55) / (1.00 - 0.55), 0.0, 1.0))
            score = int(round(80 + 10.0 * c_norm))
            score = max(80, min(90, score))

            if score >= 85:
                label = "Very strong evidence"
            else:
                label = "Strong evidence"
        else:
            # UNCERTAIN state: c ranges approximately from 0.35 to 0.55
            # Map linearly into [70, 79]
            c_norm = float(np.clip((c - 0.35) / (0.55 - 0.35), 0.0, 1.0))
            score = int(round(70 + 9.0 * c_norm))
            score = max(70, min(79, score))

            if score >= 75:
                label = "Moderate evidence"
            else:
                label = "Limited evidence"

        return {
            "score": score,
            "label": label,
            "display": f"{score} / 100",
            "is_probabilistic": False,
            "disclaimer": "Evidence strength is a presentation score derived from the calibrated model output and uncertainty measures. It is not a medical probability or diagnosis."
        }

    def process_logits(
        self,
        logits: Union[np.ndarray, List[float]]
    ) -> Dict[str, Any]:
        """
        End-to-end processing: Temperature Scaling -> Calibrated Probabilities -> Uncertainty Analysis -> Evidence Strength.
        """
        calibrated_probs = self.scaler.calibrate_logits(logits)
        metrics = self.compute_metrics(calibrated_probs)
        state, message, flags = self.evaluate_state(metrics)
        evidence = self.compute_evidence_strength(metrics, state)

        return {
            "calibrated_probabilities": calibrated_probs.tolist(),
            "metrics": metrics,
            "state": state.value,
            "evidence_strength": evidence,
            "clinical_message": message,
            "diagnostic_flags": flags,
            "temperature_used": self.scaler.temperature
        }

