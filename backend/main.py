"""
FastAPI Backend Application (Phase 14 - Production Deployment Ready)

Exposes production skin lesion classification model and Grad-CAM explainability via REST APIs.
Supports both ConvNeXt-Small champion and ResNet50 fallback architectures seamlessly.
Features:
- Lifespan model & explainability service loading
- Production CORS and security headers middleware
- Liveness and Readiness health probes
- Filename sanitization & path traversal protection
- Robust image upload validation (MIME, extension, size, PIL decode check)
- Medical/Research disclaimer integration
"""

import io
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from backend.config import (
    MODEL_ENGINE,
    MODEL_PATH,
    FALLBACK_MODEL_PATH,
    EXPECTED_MODEL_SHA256,
    MODEL_NAME,
    MODEL_ARCHITECTURE,
    MODEL_INPUT_SIZE,
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    ALLOWED_ORIGINS,
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    MEDICAL_DISCLAIMER,
    LOG_LEVEL,
    ENVIRONMENT,
    AI_PROVIDER,
    AI_API_KEY,
    AI_MODEL,
    METRICS_ENABLED,
    METRICS_AUTH_TOKEN,
)
from backend.explainability import ExplainabilityService
from backend.guidance import GuidanceRequest, GuidanceResponse, get_patient_guidance
from backend.research import get_scientific_literature
from backend.observability import (
    ObservabilityMiddleware,
    get_metrics_response,
    update_operational_gauges,
    record_inference_latency,
    record_explainability_latency,
    record_endpoint_error,
)
from src.class_labels import NUM_CLASSES, CLASS_NAMES


# Production Logger Setup
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading model and explainability service into memory on startup."""
    logger.info("=" * 60)
    logger.info(f"Starting FastAPI Backend Service (Phase 14 Deployment - Env: {ENVIRONMENT})...")
    
    model_to_load = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    logger.info(f"Loading production {MODEL_ARCHITECTURE} model from: {model_to_load}")
    
    try:
        if MODEL_ENGINE == "convnext" or str(model_to_load).endswith(".pt"):
            from src.convnext_inference import ConvNeXtLesionPredictor
            app.state.predictor = ConvNeXtLesionPredictor(model_path=model_to_load, verify_sha=True)
        else:
            from src.inference import SkinLesionPredictor
            app.state.predictor = SkinLesionPredictor(model_path=model_to_load)

        app.state.explainer = ExplainabilityService(app.state.predictor)
        app.state.startup_time = time.time()
        logger.info(f"Model & Grad-CAM Explainability service loaded successfully into memory.")
        logger.info(f"Active Architecture: {MODEL_ARCHITECTURE} | Classes: {app.state.predictor.num_classes} | Input: {app.state.predictor.input_shape}")
    except Exception as e:
        logger.error(f"Failed to load model or explainability service: {str(e)}")
        app.state.predictor = None
        app.state.explainer = None
        app.state.startup_time = time.time()

    ai_key_configured = bool(AI_API_KEY and AI_API_KEY.strip() and not AI_API_KEY.startswith("your_"))
    if ai_key_configured:
        logger.info(f"AI Guidance Layer: External LLM provider configured ({AI_PROVIDER} / {AI_MODEL}).")
    else:
        logger.info("AI Guidance Layer: Server-side AI_API_KEY not configured. Rule-based medical guidance engine ACTIVE.")

    logger.info("=" * 60)
    yield
    logger.info("Shutting down FastAPI backend service...")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Production Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Production Observability & Prometheus Telemetry Middleware
app.add_middleware(ObservabilityMiddleware)



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to return clean JSON error without internal stack traces."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    
    logger.error(f"Unhandled server error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred during request processing."},
    )


@app.get("/api/v1/health", summary="Backend Health & Readiness Check")
async def health_check(response: Response) -> Dict[str, Any]:
    """
    Returns liveness and readiness status for container orchestrators and health checks.
    Sets HTTP 503 Service Unavailable if production model is not loaded.
    """
    is_loaded = (
        hasattr(app.state, "predictor") and app.state.predictor is not None and
        hasattr(app.state, "explainer") and app.state.explainer is not None
    )
    
    if not is_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    uptime_sec = round(time.time() - getattr(app.state, "startup_time", time.time()), 2)

    # Update operational Prometheus gauges
    update_operational_gauges(
        model_loaded=is_loaded,
        explainability_loaded=is_loaded,
        uptime_seconds=uptime_sec,
        model_name=MODEL_NAME,
        architecture=MODEL_ARCHITECTURE,
        checkpoint_sha256=EXPECTED_MODEL_SHA256,
        guidance_provider=AI_PROVIDER if (AI_API_KEY and not AI_API_KEY.startswith("your_")) else "rule_fallback",
        guidance_ok=True,
    )

    return {
        "status": "healthy" if is_loaded else "degraded",
        "liveness": "alive",
        "readiness": "ready" if is_loaded else "not_ready",
        "model": MODEL_NAME,
        "model_version": "v1",
        "architecture": MODEL_ARCHITECTURE,
        "model_loaded": is_loaded,
        "explainability_loaded": is_loaded,
        "uptime_seconds": uptime_sec,
    }



@app.get("/api/v1/model", summary="Model Metadata Information")
async def get_model_info() -> Dict[str, Any]:
    """Return active production model specifications and canonical class metadata."""
    return {
        "model_name": MODEL_NAME,
        "version": "v1",
        "architecture": MODEL_ARCHITECTURE,
        "num_classes": NUM_CLASSES,
        "input_size": MODEL_INPUT_SIZE,
        "status": "production_active",
        "disclaimer": MEDICAL_DISCLAIMER,
    }


async def _validate_and_read_image(image: UploadFile) -> Image.Image:
    """
    Helper to sanitize input filenames, validate file headers, extensions, payload limits,
    and decode image into PIL Image.
    """
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file attached to request.",
        )

    # Sanitize filename (strip directory traversal path elements)
    sanitized_filename = Path(image.filename).name
    file_ext = Path(sanitized_filename).suffix.lower()

    if not file_ext or file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension '{file_ext}'. Allowed formats: JPEG, JPG, PNG, WEBP.",
        )

    try:
        contents = await image.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file contents.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {max_mb}MB.",
        )

    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()  # Verify image header integrity
        pil_img = Image.open(io.BytesIO(contents))  # Re-open after verify
        return pil_img
    except (UnidentifiedImageError, Exception):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupt or invalid image file. Unable to decode image contents.",
        )


@app.post("/api/v1/predict", summary="Predict Skin Lesion Class from Image")
async def predict(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts multipart image upload, validates image integrity, evaluates quality gate,
    and runs calibrated inference with uncertainty estimation.
    """
    if not hasattr(app.state, "predictor") or app.state.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or failed to initialize.",
        )

    pil_img = await _validate_and_read_image(image)

    start_time = time.perf_counter()
    
    # 1. Pre-inference Image Quality Gate Check
    quality_report = app.state.predictor.quality_gate.analyze(pil_img)
    if quality_report["state"] == "QUALITY_FAIL":
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "success": False,
            "quality": quality_report,
            "prediction": None,
            "top3": [],
            "uncertainty": {
                "state": "ABSTAIN",
                "clinical_message": quality_report["guidance_message"]
            },
            "evidence_strength": None,
            "evidence_label": "Image quality is insufficient for reliable analysis.",
            "evidence_display": "N/A",
            "latency_ms": round(latency_ms, 2),
            "detail": quality_report["guidance_message"],
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    # 2. Model Inference with Temperature Scaling & Uncertainty Analysis
    try:
        inf_start = time.perf_counter()
        res = app.state.predictor.predict(pil_img)
        inf_duration = time.perf_counter() - inf_start
        record_inference_latency(MODEL_ARCHITECTURE, inf_duration)
    except Exception as e:
        record_endpoint_error("/api/v1/predict", "inference_error")
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed while processing image.",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    prediction_payload = {
        "class": res["predicted_class"],
        "class_index": res["predicted_index"],
        "calibrated_probability": float(res["confidence"]),
        "probability": float(res["confidence"]),
        "confidence": float(res["confidence"]),
        "raw_confidence": float(res["raw_confidence"]),
        "uncertainty_state": res["uncertainty"]["state"],
        "margin": float(res["uncertainty"]["metrics"]["prediction_margin"]),
        "normalized_entropy": float(res["uncertainty"]["metrics"]["normalized_entropy"]),
        "is_calibrated": res["is_calibrated"],
        "temperature_used": res["temperature_used"]
    }

    return {
        "success": True,
        "quality": quality_report,
        "prediction": prediction_payload,
        "top3": [
            {
                "class": item["class"],
                "class_index": item["index"],
                "calibrated_probability": float(item["confidence"]),
                "probability": float(item["confidence"]),
                "confidence": float(item["confidence"]),
                "raw_confidence": float(item.get("raw_confidence", item["confidence"])),
            }
            for item in res["top3"]
        ],
        "uncertainty": res["uncertainty"],
        "model": {
            "name": MODEL_NAME,
            "version": "v1",
            "architecture": MODEL_ARCHITECTURE,
            "sha256": EXPECTED_MODEL_SHA256
        },
        "latency_ms": round(latency_ms, 2),
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@app.post("/api/v1/explain", summary="Generate Grad-CAM Explainability Heatmap & Overlay")
async def explain(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts multipart image upload, evaluates quality gate, and computes
    Grad-CAM spatial activation heatmaps and overlays for the predicted class.
    """
    if not hasattr(app.state, "explainer") or app.state.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainability service is currently unavailable.",
        )

    pil_img = await _validate_and_read_image(image)

    start_time = time.perf_counter()
    
    # Pre-inference Quality Check
    quality_report = app.state.predictor.quality_gate.analyze(pil_img)
    if quality_report["state"] == "QUALITY_FAIL":
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "success": False,
            "quality": quality_report,
            "detail": quality_report["guidance_message"],
            "latency_ms": round(latency_ms, 2),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    try:
        exp_start = time.perf_counter()
        explanation = app.state.explainer.generate_explanation(pil_img)
        exp_duration = time.perf_counter() - exp_start
        record_explainability_latency(MODEL_ARCHITECTURE, exp_duration)
    except Exception as e:
        record_endpoint_error("/api/v1/explain", "gradcam_error")
        logger.error(f"Grad-CAM generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grad-CAM generation failed while processing image.",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    explanation["success"] = True
    explanation["quality"] = quality_report
    explanation["latency_ms"] = round(latency_ms, 2)
    return explanation


@app.get("/api/v1/ai/status", summary="AI Patient Guidance Service Readiness")
async def get_ai_status() -> Dict[str, Any]:
    """
    Returns AI guidance layer readiness status, provider configuration,
    and canonical 14 trained classes list.
    """
    key_configured = bool(AI_API_KEY and AI_API_KEY.strip() and not AI_API_KEY.startswith("your_"))
    return {
        "available": True,
        "provider": AI_PROVIDER if key_configured else "rule_fallback_engine",
        "model": AI_MODEL if key_configured else "medical_rule_v1",
        "mode": "llm" if key_configured else "rule_fallback",
        "supported_classes": CLASS_NAMES,
    }


@app.post("/api/v1/guidance", summary="AI Patient Educational Guidance")
async def generate_guidance(request: GuidanceRequest) -> GuidanceResponse:
    """
    Generates non-diagnostic patient education, triage urgency, red flag detection,
    and self-care guidance for skin lesion analysis results.
    """
    try:
        response = await get_patient_guidance(request)
        return response
    except Exception as e:
        logger.error(f"Error in guidance service endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The educational guidance service encountered an unexpected error. Your prediction result remains available.",
        )


@app.get("/api/v1/research/{condition}", summary="Biomedical Literature Research Search")
async def get_research_literature(condition: str) -> Dict[str, Any]:
    """
    Returns research literature link and metadata for Google Scholar
    for a given predicted skin condition category.
    """
    try:
        return await get_scientific_literature(condition)
    except Exception as e:
        logger.error(f"Error in research literature endpoint: {str(e)}")
        return {
            "success": False,
            "condition": condition,
            "source": "fallback",
            "articles": [],
            "message": "Scientific literature service is temporarily unavailable. Prediction and Grad-CAM remain fully functional."
        }


@app.get("/metrics", summary="Prometheus Operational Metrics", include_in_schema=False)
async def metrics(request: Request):
    """
    Exposes Prometheus-compatible operational metrics.
    Protected by optional Bearer token if METRICS_AUTH_TOKEN is configured.
    """
    if not METRICS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics exporter endpoint is disabled.",
        )
    auth_header = request.headers.get("Authorization")
    return get_metrics_response(auth_header=auth_header, required_token=METRICS_AUTH_TOKEN if METRICS_AUTH_TOKEN else None)


if __name__ == "__main__":

    import uvicorn
    from backend.config import API_HOST, API_PORT
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=False)
