FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency file
COPY pyproject.toml ./

# Install torch first (stable base)
RUN uv pip install --system --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy project
COPY . .

# Install project deps
RUN uv pip install --system --no-cache-dir .

# 🔴 FORCE numpy LAST
RUN uv pip install --system --no-cache-dir "numpy<2"

# Run API
CMD ["metaextractor", "--api"]