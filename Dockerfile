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

# Offline-first: the embedding weights are baked into HF_HOME below.
# HOME=/tmp: `make up` runs the container as the invoking user on rootful Docker
# (so files it writes into the bind-mounted demo_app stay that user's); such a
# uid has no home directory in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    HOME=/tmp \
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

# Install PyTorch CPU and all required Python packages. p3/requirements.txt
# carries flake8: demo_app/ioc.config.yml runs `python3 -m flake8`, and without
# it every validation in the container failed with "No module named flake8".
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        -r /app/p1/requirements.txt \
        -r /app/p2/requirements.txt \
        -r /app/p3/requirements.txt \
        -r /app/bonus/requirements.txt \
        uvicorn pillow && \
    python3 -m flake8 --version

# Bake the embedding weights into the image. HF_HUB_OFFLINE=1 is set above, so
# without this layer the container can only start when a host cache happens to be
# mounted at /tmp/ioc/hf-cache - which makes `make up` fail on a clean machine and
# after `make fclean`. HF_HUB_OFFLINE is lifted for this one RUN only. The cache
# is then made readable by any uid, since the container may not run as root.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('all-MiniLM-L6-v2'); \
     print('[+] embedding weights baked into image')" && \
    chmod -R a+rX /tmp/ioc/hf-cache

# Copy source trees and demo application
COPY p1/ /app/p1/
COPY p2/ /app/p2/
COPY p3/ /app/p3/
COPY bonus/ /app/bonus/
COPY demo_app/ /app/demo_app/
COPY Makefile /app/Makefile

# Expose the API and unified 4-tab web dashboard port
EXPOSE 8000

# Default command (plain `docker run -p 8000:8000`): same suite as compose, on
# all interfaces. docker-compose.yml overrides it for host networking.
CMD ["python3", "bonus/index.py", "demo_app", "--dashboard", "--watch", "--host", "0.0.0.0", "--port", "8000", "--db-dir", "/tmp/ioc/chroma_db"]
