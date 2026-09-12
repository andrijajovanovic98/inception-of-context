# Inception-of-Context (IoC) - Production Dockerfile
# Provides a clean, isolated Python 3.10 environment compliant with Subject Chapter IV & VIII.

FROM python:3.10-slim

# Install minimal OS dependencies for Git operations, compilation, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Offline-first: embedding weights must exist under HF_HOME (mounted from host
# /tmp/ioc/hf-cache produced by `make setup`, or baked in at build time).
ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    HF_HOME=/tmp/ioc/hf-cache \
    TRANSFORMERS_CACHE=/tmp/ioc/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OLLAMA_HOST=http://host.docker.internal:11435

# Copy all requirements files first for efficient Docker layer caching
COPY p1/requirements.txt /app/p1/requirements.txt
COPY p2/requirements.txt /app/p2/requirements.txt
COPY p3/requirements.txt /app/p3/requirements.txt
COPY bonus/requirements.txt /app/bonus/requirements.txt

# Install PyTorch CPU and all required Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        -r /app/p1/requirements.txt \
        -r /app/p2/requirements.txt \
        -r /app/p3/requirements.txt \
        -r /app/bonus/requirements.txt \
        uvicorn pillow

# Copy source trees and demo application
COPY p1/ /app/p1/
COPY p2/ /app/p2/
COPY p3/ /app/p3/
COPY bonus/ /app/bonus/
COPY demo_app/ /app/demo_app/
COPY Makefile /app/Makefile

# Expose the API and unified 4-tab web dashboard port
EXPOSE 8000

# Default command: launch the unified Bonus suite & Dashboard over demo_app
CMD ["python3", "bonus/index.py", "demo_app", "--dashboard", "--host", "0.0.0.0", "--port", "8000"]

