"""
Backend Production Configuration Module (Phase 14 Production)

Loads environment variables and sets sensible defaults for production backend API operation.
Uses project-relative paths to avoid machine-specific absolute path dependencies.
Supports production environment overrides without exposing hardcoded secrets.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Base Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from backend/.env or root .env
load_dotenv(PROJECT_ROOT / "backend" / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# Environment & Server Settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

# Model Configuration — ConvNeXt-Small Production Champion (Phase 14)
MODEL_ENGINE = os.getenv("MODEL_ENGINE", "convnext").lower()  # "convnext" or "resnet50"
DEFAULT_CONVNEXT_PATH = PROJECT_ROOT / "models" / "production" / "convnext_small_skin_lesion_v1.pt"
DEFAULT_RESNET_PATH = PROJECT_ROOT / "models" / "production" / "resnet50_skin_lesion_v1.keras"

if MODEL_ENGINE == "convnext":
    MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_CONVNEXT_PATH))
    FALLBACK_MODEL_PATH = Path(PROJECT_ROOT / "models" / "experiments" / "phase10a_convnext" / "convnext_small_best.pt")
    EXPECTED_MODEL_SHA256 = "67f4d8d04edef9389aeac77688dcc90e193b717de67f8c4c0c1ebf15483a3fe3"
    MODEL_NAME = "convnext_small_v1"
    MODEL_ARCHITECTURE = "ConvNeXt-Small"
    MODEL_INPUT_SIZE = "384x384"
else:
    MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_RESNET_PATH))
    FALLBACK_MODEL_PATH = Path(PROJECT_ROOT / "models" / "resnet50_phase26_best.keras")
    EXPECTED_MODEL_SHA256 = "93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e"
    MODEL_NAME = "resnet50_skin_lesion_v1"
    MODEL_ARCHITECTURE = "ResNet50"
    MODEL_INPUT_SIZE = "224x224"

# AI Patient Guidance Assistant Configuration (Server-Side)
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini")
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gemini-1.5-flash")

# Observability & Metrics Configuration (Phase 16)
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() in ("true", "1", "yes")
METRICS_AUTH_TOKEN = os.getenv("METRICS_AUTH_TOKEN", "")


# Upload Configuration (15MB max upload size default)
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE", 15 * 1024 * 1024))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# CORS Configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "")
DEFAULT_DEVELOPMENT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:8002",
    "http://127.0.0.1:8002",
]
if ALLOWED_ORIGINS_RAW:
    ALLOWED_ORIGINS: List[str] = [
        origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",") if origin.strip()
    ]
else:
    if ENVIRONMENT == "production":
        # Phase 22A Security Hardening: Fail-safe in production by disabling permissive dev origins.
        # An unset production allowlist must fail safely (empty allowlist, blocking cross-origin requests)
        # rather than silently enabling permissive development origins.
        ALLOWED_ORIGINS: List[str] = []
    else:
        ALLOWED_ORIGINS: List[str] = DEFAULT_DEVELOPMENT_ORIGINS



# API Metadata
API_TITLE = "Skin Lesion Classification & Grad-CAM API"
API_VERSION = "v1"
API_DESCRIPTION = (
    f"FastAPI production backend for {MODEL_ARCHITECTURE} skin lesion classification and Grad-CAM explainability. "
    "Designed for research and educational demonstration. NOT a medical diagnostic tool."
)
MEDICAL_DISCLAIMER = (
    "This tool provides an AI model prediction for research/educational purposes "
    "and is not a medical diagnosis. Consult a board-certified dermatologist for medical evaluation."
)
