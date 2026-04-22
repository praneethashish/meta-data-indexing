# Use python-slim for a smaller base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GEMMA_MODEL_PATH=/app/models/gemma-4-E4B-it-Q4_K_M.gguf

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only requirements first to leverage Docker cache
COPY pyproject.toml .

# Install dependencies
# We explicitly install the CPU version of PyTorch to keep the image slim
RUN uv pip install --system --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system --no-cache-dir opencv-python-headless \
    && uv pip install --system --no-cache-dir .

# Copy the rest of the application
COPY . .

# Final installation of the package in editable mode or normally
RUN uv pip install --system --no-cache-dir .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uv", "run", "bookextractor", "--api"]
