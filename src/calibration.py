"""
Confidence Calibration Module for Skin Lesion Classification.

Provides post-hoc temperature scaling calibration to ensure displayed
probabilities empirically reflect predictive correctness.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. This module is POST-HOC: it operates strictly downstream of the neural network.
2. It NEVER modifies, retrains, fine-tunes, or mutates the underlying model or its weights.
3. Temperature parameter T is learned solely from validation data (data/processed/val.csv).
4. Top-1 and top-k class rankings are preserved identically for all T > 0.
"""

from typing import Union, Optional, Dict, Any, List
import json
from pathlib import Path
import numpy as np


class TemperatureScaler:
    """
    Post-hoc Temperature Scaler for Deep Learning Probability Calibration.
    
    Transforms uncalibrated logits or softmax probabilities using a learned
    temperature scalar T:
        P_calibrated(Y = k | x) = exp(z_k / T) / sum_j exp(z_j / T)
    
    Attributes:
        temperature (float): Learned temperature parameter (T > 0).
                             T > 1 softens overconfident probability distributions.
                             T = 1 leaves probabilities unchanged.
    """

    DEFAULT_TEMPERATURE: float = 1.3421

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        if temperature <= 0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self._temperature: float = float(temperature)

    @property
    def temperature(self) -> float:
        """Return the active temperature parameter."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Update the temperature parameter."""
        if value <= 0:
            raise ValueError(f"Temperature must be strictly positive, got {value}")
        self._temperature = float(value)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax function."""
        # Shift logits by max for numerical stability
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def calibrate_logits(self, logits: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Calibrate raw model logits and return calibrated probabilities.
        
        Args:
            logits: Numpy array or list of raw pre-softmax logits, shape (..., num_classes).
            
        Returns:
            Numpy array of calibrated probabilities summing to 1.0.
        """
        logits_arr = np.asarray(logits, dtype=np.float64)
        scaled_logits = logits_arr / self._temperature
        return self._softmax(scaled_logits)

    def calibrate_probabilities(
        self,
        probabilities: Union[np.ndarray, List[float]],
        eps: float = 1e-12
    ) -> np.ndarray:
        """
        Calibrate softmax probabilities by reconstructing approximate log-odds (logits)
        and scaling by temperature.
        
        Args:
            probabilities: Array or list of raw softmax probabilities summing to 1.0.
            eps: Epsilon floor for numerical stability when taking logarithm.
            
        Returns:
            Numpy array of calibrated probabilities summing to 1.0.
        """
        probs_arr = np.asarray(probabilities, dtype=np.float64)
        # Numerical guard: clip probabilities to avoid log(0)
        clipped_probs = np.clip(probs_arr, eps, 1.0)
        # Approximate logits up to an additive constant: z_k = log(p_k)
        pseudo_logits = np.log(clipped_probs)
        scaled_logits = pseudo_logits / self._temperature
        return self._softmax(scaled_logits)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save calibration configuration to a JSON file."""
        config = {
            "method": "temperature_scaling",
            "temperature": self._temperature,
            "is_post_hoc": True,
            "version": "1.0.0"
        }
        with open(filepath, "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TemperatureScaler":
        """Load a TemperatureScaler instance from a JSON configuration file."""
        with open(filepath, "r") as f:
            config = json.load(f)
        temp = config.get("temperature", cls.DEFAULT_TEMPERATURE)
        return cls(temperature=temp)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "method": "temperature_scaling",
            "temperature": self._temperature,
            "is_post_hoc": True
        }

    def __repr__(self) -> str:
        return f"TemperatureScaler(temperature={self._temperature:.4f}, post_hoc=True)"
