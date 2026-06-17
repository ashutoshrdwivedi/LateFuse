# Multi-stage Dockerfile for LoRA multi-tenant inference server
# Designed for DigitalOcean Managed Kubernetes with GPU droplets

# =============================================================================
# Stage 1: Builder — install dependencies in a clean layer
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install server dependencies first (better layer caching)
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

# Install the lora-serving package
COPY pyproject.toml /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -e .

# Copy deploy server code
COPY deploy/ /app/deploy/

# =============================================================================
# Stage 2: Runtime — bake in the base model for fast cold starts
# =============================================================================
FROM builder AS runtime

# Pre-download the HuggingFace base model at build time.
# This avoids a 30-60s download on every pod startup.
ARG MODEL_NAME=intfloat/multilingual-e5-small
RUN python3.10 -c "\
from transformers import AutoModel, AutoTokenizer, AutoConfig; \
AutoConfig.from_pretrained('${MODEL_NAME}'); \
AutoModel.from_pretrained('${MODEL_NAME}'); \
AutoTokenizer.from_pretrained('${MODEL_NAME}'); \
print('Model cached successfully')"

# Create adapter cache directory
RUN mkdir -p /tmp/adapters

EXPOSE 8080

# Health check for Docker (K8s uses its own probes)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3.10 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run with 1 worker (GPU inference is single-process)
CMD ["python3.10", "-m", "uvicorn", "deploy.server.app:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1", \
     "--log-level", "info"]
