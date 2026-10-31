"""
Backend Production Observability & Monitoring Module (Phase 16)

Provides a lightweight, zero-overhead Prometheus-compatible metrics exporter,
structured logging middleware, bounded-cardinality telemetry, and error tracking.

Features:
- Prometheus metrics exporter (/metrics endpoint)
- Bounded-cardinality metric labels (zero unbounded dynamic user strings)
- Non-blocking telemetry (monitoring failures NEVER interrupt inference)
- Privacy & security protection (zero patient images, auth keys, or PII logged)
- Structured JSON-format operational logs
- Process resource gauges (CPU & Resident Memory)
"""

import time
import logging
import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse
from fastapi import HTTPException, status

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Fallback psutil / resource
try:
    import psutil
except ImportError:
    psutil = None
import resource

logger = logging.getLogger("backend.observability")

# -----------------------------------------------------------------------------
# Metric Registry & Definitions (Bounded Cardinality)
# -----------------------------------------------------------------------------

# Recognized endpoint categories (strictly bounded)
ALLOWED_ENDPOINT_CATEGORIES = {
    "/api/v1/health": "health",
    "/api/v1/model": "model",
    "/api/v1/predict": "predict",
    "/api/v1/explain": "explain",
    "/api/v1/ai/status": "ai_status",
    "/api/v1/guidance": "guidance",
    "/metrics": "metrics",
}

def classify_endpoint(path: str) -> str:
    """Map arbitrary request paths to a strictly bounded set of categories."""
    if path in ALLOWED_ENDPOINT_CATEGORIES:
        return ALLOWED_ENDPOINT_CATEGORIES[path]
    if path.startswith("/api/v1/research/"):
        return "research"
    if path.startswith("/docs") or path.startswith("/redoc") or path == "/openapi.json":
        return "docs"
    return "other"

def classify_status_class(status_code: int) -> str:
    """Return status code class for bounded cardinality."""
    if 200 <= status_code < 300:
        return "2xx"
    elif 300 <= status_code < 400:
        return "3xx"
    elif 400 <= status_code < 500:
        return "4xx"
    elif 500 <= status_code < 600:
        return "5xx"
    return "unknown"


if PROMETHEUS_AVAILABLE:
    # 1. Request Counters
    REQUESTS_TOTAL = Counter(
        "skin_lesion_http_requests_total",
        "Total HTTP requests received by endpoint and status category",
        ["endpoint", "method", "status_class"],
    )

    # 2. HTTP Request Latency Histogram
    REQUEST_DURATION_SECONDS = Histogram(
        "skin_lesion_http_request_duration_seconds",
        "Total HTTP request latency in seconds by endpoint",
        ["endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    # 3. Model Inference Latency Histogram
    INFERENCE_DURATION_SECONDS = Histogram(
        "skin_lesion_inference_duration_seconds",
        "Pure forward inference duration in seconds",
        ["architecture"],
        buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0, 2.0],
    )

    # 4. Explainability (Grad-CAM) Latency Histogram
    EXPLAINABILITY_DURATION_SECONDS = Histogram(
        "skin_lesion_explainability_duration_seconds",
        "Grad-CAM calculation and heatmap rendering duration in seconds",
        ["architecture"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    )

    # 5. Endpoint Error Counter
    ENDPOINT_ERRORS_TOTAL = Counter(
        "skin_lesion_endpoint_errors_total",
        "Total errors encountered by endpoint and sanitized error category",
        ["endpoint", "error_type"],
    )

    # 6. Service Health & Load Gauges
    MODEL_LOADED_GAUGE = Gauge(
        "skin_lesion_model_loaded",
        "Indicates if skin lesion classification model is loaded (1=yes, 0=no)",
        ["model_name", "architecture", "checkpoint_sha256"],
    )

    EXPLAINABILITY_LOADED_GAUGE = Gauge(
        "skin_lesion_explainability_loaded",
        "Indicates if Grad-CAM explainability service is loaded (1=yes, 0=no)",
    )

    SERVICE_UPTIME_SECONDS_GAUGE = Gauge(
        "skin_lesion_service_uptime_seconds",
        "Service uptime in seconds since process startup",
    )

    EXTERNAL_PROVIDER_GAUGE = Gauge(
        "skin_lesion_external_provider_status",
        "Operational readiness of external guidance and research providers (1=ready, 0=degraded)",
        ["provider_name"],
    )

    # 7. System Resource Gauges
    PROCESS_RESIDENT_MEMORY_BYTES = Gauge(
        "skin_lesion_process_resident_memory_bytes",
        "Resident memory size of the inference service process in bytes",
    )

    PROCESS_CPU_PERCENT = Gauge(
        "skin_lesion_process_cpu_percent",
        "Current CPU utilization percentage of the inference service process",
    )
else:
    REQUESTS_TOTAL = None
    REQUEST_DURATION_SECONDS = None
    INFERENCE_DURATION_SECONDS = None
    EXPLAINABILITY_DURATION_SECONDS = None
    ENDPOINT_ERRORS_TOTAL = None
    MODEL_LOADED_GAUGE = None
    EXPLAINABILITY_LOADED_GAUGE = None
    SERVICE_UPTIME_SECONDS_GAUGE = None
    EXTERNAL_PROVIDER_GAUGE = None
    PROCESS_RESIDENT_MEMORY_BYTES = None
    PROCESS_CPU_PERCENT = None


# -----------------------------------------------------------------------------
# Structured Logger
# -----------------------------------------------------------------------------

def log_structured_access(
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: float,
    client_host: str = "internal",
    error_summary: Optional[str] = None,
):
    """
    Emits a structured JSON access log entry.
    Strictly scrubs request payloads, image data, and auth headers.
    """
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "type": "access",
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "client": client_host if client_host in ("127.0.0.1", "localhost", "internal") else "external",
    }
    if error_summary:
        entry["error"] = error_summary

    log_msg = json.dumps(entry)
    if status_code >= 500:
        logger.error(log_msg)
    elif status_code >= 400:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)


# -----------------------------------------------------------------------------
# Telemetry Recording Helpers
# -----------------------------------------------------------------------------

def record_inference_latency(architecture: str, duration_seconds: float):
    """Record pure model inference latency in histogram safely."""
    try:
        if PROMETHEUS_AVAILABLE and INFERENCE_DURATION_SECONDS:
            INFERENCE_DURATION_SECONDS.labels(architecture=architecture).observe(duration_seconds)
    except Exception as e:
        logger.debug(f"Telemetry recording skipped: {e}")

def record_explainability_latency(architecture: str, duration_seconds: float):
    """Record Grad-CAM generation latency in histogram safely."""
    try:
        if PROMETHEUS_AVAILABLE and EXPLAINABILITY_DURATION_SECONDS:
            EXPLAINABILITY_DURATION_SECONDS.labels(architecture=architecture).observe(duration_seconds)
    except Exception as e:
        logger.debug(f"Telemetry recording skipped: {e}")

def record_endpoint_error(endpoint: str, error_type: str):
    """Record endpoint error safely with bounded error_type."""
    try:
        if PROMETHEUS_AVAILABLE and ENDPOINT_ERRORS_TOTAL:
            category = classify_endpoint(endpoint)
            ENDPOINT_ERRORS_TOTAL.labels(endpoint=category, error_type=error_type).inc()
    except Exception as e:
        logger.debug(f"Telemetry recording skipped: {e}")

def update_operational_gauges(
    model_loaded: bool,
    explainability_loaded: bool,
    uptime_seconds: float,
    model_name: str,
    architecture: str,
    checkpoint_sha256: str,
    guidance_provider: str = "rule_fallback",
    guidance_ok: bool = True,
):
    """Update state gauges and system resources during health/metric scrapes."""
    try:
        if not PROMETHEUS_AVAILABLE:
            return

        # 1. State Gauges
        if MODEL_LOADED_GAUGE:
            # First reset or update
            MODEL_LOADED_GAUGE.labels(
                model_name=model_name,
                architecture=architecture,
                checkpoint_sha256=checkpoint_sha256[:16] if checkpoint_sha256 else "none",
            ).set(1.0 if model_loaded else 0.0)

        if EXPLAINABILITY_LOADED_GAUGE:
            EXPLAINABILITY_LOADED_GAUGE.set(1.0 if explainability_loaded else 0.0)

        if SERVICE_UPTIME_SECONDS_GAUGE:
            SERVICE_UPTIME_SECONDS_GAUGE.set(uptime_seconds)

        if EXTERNAL_PROVIDER_GAUGE:
            EXTERNAL_PROVIDER_GAUGE.labels(provider_name=guidance_provider).set(1.0 if guidance_ok else 0.0)

        # 2. System Resource Gauges
        if psutil is not None:
            proc = psutil.Process()
            mem_bytes = proc.memory_info().rss
            cpu_pct = proc.cpu_percent(interval=None)
            if PROCESS_RESIDENT_MEMORY_BYTES:
                PROCESS_RESIDENT_MEMORY_BYTES.set(mem_bytes)
            if PROCESS_CPU_PERCENT:
                PROCESS_CPU_PERCENT.set(cpu_pct)
        else:
            # Fallback to resource
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS ru_maxrss is in bytes, Linux is in KB
            if PROCESS_RESIDENT_MEMORY_BYTES:
                PROCESS_RESIDENT_MEMORY_BYTES.set(rss_bytes)
            if PROCESS_CPU_PERCENT:
                PROCESS_CPU_PERCENT.set(0.0)

    except Exception as e:
        logger.debug(f"Operational gauges update skipped: {e}")


# -----------------------------------------------------------------------------
# Observability Middleware
# -----------------------------------------------------------------------------

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware tracking HTTP request rate, bounded status codes,
    duration histograms, and emitting sanitized structured access logs.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        endpoint_category = classify_endpoint(request.url.path)
        method = request.method

        # Do not log or monitor CORS OPTIONS preflights heavily
        if method == "OPTIONS":
            return await call_next(request)

        status_code = 500
        error_summary = None

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        except HTTPException as he:
            status_code = he.status_code
            error_summary = str(he.detail) if isinstance(he.detail, str) else "http_exception"
            raise he
        except Exception as exc:
            status_code = 500
            error_summary = f"{type(exc).__name__}: {str(exc)[:100]}"
            raise exc
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            status_class = classify_status_class(status_code)

            # Safely record Prometheus metrics
            try:
                if PROMETHEUS_AVAILABLE and REQUESTS_TOTAL and REQUEST_DURATION_SECONDS:
                    REQUESTS_TOTAL.labels(
                        endpoint=endpoint_category,
                        method=method,
                        status_class=status_class,
                    ).inc()
                    REQUEST_DURATION_SECONDS.labels(
                        endpoint=endpoint_category,
                    ).observe(elapsed_seconds)
            except Exception as e:
                logger.debug(f"Metrics update exception: {e}")

            # Structured logging
            client_host = request.client.host if request.client else "internal"
            log_structured_access(
                endpoint=request.url.path,
                method=method,
                status_code=status_code,
                latency_ms=elapsed_seconds * 1000.0,
                client_host=client_host,
                error_summary=error_summary,
            )


# -----------------------------------------------------------------------------
# Prometheus Exporter Handler
# -----------------------------------------------------------------------------

def get_metrics_response(auth_header: Optional[str] = None, required_token: Optional[str] = None) -> Response:
    """
    Generates standard Prometheus metric output.
    Supports bearer token check.
    Enforces restricted access (HTTP 403) when METRICS_AUTH_TOKEN is not configured.
    """
    # Phase 22A Security Hardening: Ensure /metrics is not publicly accessible without configured token.
    # If token-based authentication is not configured, enforce restriction (HTTP 403 Forbidden).
    if not required_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to metrics is restricted. METRICS_AUTH_TOKEN is not configured on this server.",
        )

    # Check bearer token
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to view system metrics.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header.split(" ", 1)[1].strip()
    if token != required_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid metrics access token.",
        )

    if not PROMETHEUS_AVAILABLE:
        return PlainTextResponse(
            "# Prometheus metrics exporter not available: prometheus-client not installed.\n",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    output = generate_latest()
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)


