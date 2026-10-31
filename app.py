"""Skin Lesion Classification Prototype Streamlit Web Application.

Run locally via:
    streamlit run app.py
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.data_pipeline import LABEL_MAP, NUM_CLASSES
from src.model import build_model

# Standard human-readable class names
FULL_CLASS_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_css(css_path: Path) -> None:
    """Load custom vanilla CSS stylesheet into Streamlit."""
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


