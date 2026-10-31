# ==============================================================================
# DermaVision AI — Production Backend Dockerfile (Phase 32)
# Production container specification for frozen ResNet50 FastAPI service
# ==============================================================================

FROM python:3.11-slim as base

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    PORT=8000

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user for container security
RUN useradd -m -u 1000 appuser

# Copy backend requirements first (caching layer)
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application code & frozen production model artifacts
COPY backend /app/backend
COPY src /app/src
COPY models/production /app/models/production

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Container Healthcheck targeting FastAPI liveness/readiness endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start production Uvicorn ASGI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
