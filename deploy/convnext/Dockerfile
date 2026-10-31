# Production Dockerfile for Isolated ConvNeXt-Small Deployment (Phase 13)
# Base image: Official Python 3.11 Slim
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY deploy/convnext/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and verified model weights
COPY backend/ /app/backend/
COPY src/ /app/src/
COPY models/experiments/phase10a_convnext/convnext_small_best.pt /app/models/experiments/phase10a_convnext/convnext_small_best.pt

# Expose port
EXPOSE 8002

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_PORT=8002 \
    API_HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/api/v1/health || exit 1

# Run entry point
CMD ["python", "-m", "uvicorn", "backend.convnext_app:app", "--host", "0.0.0.0", "--port", "8002"]
