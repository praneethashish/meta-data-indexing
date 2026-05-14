# BookExtractor

A production-ready Python package for extracting structured metadata from scanned Telugu, Hindi, and English books, magazines, and images.

## Overview

`bookextractor` uses a multi-stage pipeline combining vParse OCR, local VLM inference (vLLM), and rule-based validation to extract structured metadata from noisy, layout-agnostic scans. Supports both synchronous CLI and asynchronous API with Celery/Redis background processing.

## Key Features

- **Multi-format support**: PDF, images (JPG, PNG, WebP, TIFF), text files (MD, JSON, TXT)
- **vParse OCR integration**: Multilingual OCR via mineru-dots API (English, Telugu, Hindi)
- **Local VLM inference**: Qwen2.5-VL, Qwen3-VL, Gemma 4, and more via vLLM — data never leaves your machine
- **Async task queue**: Celery + Redis for background processing with GPU/CPU worker separation
- **Auto hardware detection**: Optimizes vLLM config for NVIDIA GPU, TPU, Apple Silicon, or CPU
- **Magazine detection**: Auto-detects magazine vs book content with specialized extraction prompts
- **ISBN validation**: Regex extraction + checksum validation + Open Library API lookup (with retry)
- **Model management**: Interactive CLI for downloading, listing, and removing cached models
- **Production hardening**: Lifecycle-managed VLM, upload size limits, safe file cleanup, retry logic

## Installation

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd Meta-Data-Indexing

# Install dependencies and package
uv pip install -e .

# Set up Git hooks (mandatory for contributors)
uv run pre-commit install
```

## Usage

### CLI

```bash
# Extract metadata from a file
uv run bookextractor extract input.pdf output.json

# Specify OCR language
uv run bookextractor extract input.pdf output.json --lang te
uv run bookextractor extract input.pdf output.json --lang devanagari

# Benchmark mode (includes debug info)
uv run bookextractor extract input.pdf output.json --benchmark

# Reduce VRAM usage
uv run bookextractor extract input.pdf output.json --max-model-len 2048
```

### Model Management

```bash
# List available models and cache status
uv run bookextractor model list

# Interactively download models
uv run bookextractor model download

# Remove a cached model
uv run bookextractor model remove <model-id>

# Show cache disk usage
uv run bookextractor model cache
```

### Hardware Detection

```bash
uv run bookextractor hardware-info
```

### API (FastAPI)

```bash
# Start the web server
uv run bookextractor api
uv run bookextractor api --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/extract` | POST | Synchronous extraction (images only, max 500MB) |
| `/extract/async` | POST | Submit async job, returns `job_id` |
| `/jobs/{job_id}` | GET | Poll job status/result |

**Swagger docs**: `http://localhost:8000/docs`

### Celery Worker

```bash
# Start a worker for a specific queue
uv run bookextractor worker --queue vlm_queue --concurrency 2
uv run bookextractor worker --queue default_queue --concurrency 8
```

## Docker

### Quick Start

```bash
docker compose up --build -d
```

This starts:
- **bookextractor** — FastAPI service (port 8000)
- **redis** — Message broker and result backend (port 6379)
- **worker-gpu** — GPU worker for PDF/text extraction (vlm_queue)
- **worker-cpu** — CPU worker for image extraction (default_queue)
- **model-downloader** — Downloads the selected model to shared cache

### Select a Model

```bash
VLLM_MODEL_ID=google/gemma-4-E4B-it docker compose up --build -d
```

### VParse OCR (Optional)

```bash
# Pipeline mode (CPU/GPU)
docker compose --profile pipeline up --build -d

# VLM mode (GPU recommended)
docker compose --profile vlm up --build -d

# Hybrid mode
docker compose --profile hybrid up --build -d
```

| Service | Host Port | Description |
|---------|-----------|-------------|
| `bookextractor` | 8000 | FastAPI + LLM |
| `redis` | 6379 | Message broker |
| `vparse` | 9000 | OCR API (optional) |

### Model Persistence

Models are stored in `~/.cache/huggingface` on the host and mounted into all containers. They persist across restarts and rebuilds.

```bash
docker compose logs -f model-downloader
```

## Architecture

```
Client → FastAPI (lifespan-managed VLM) → Redis Broker → Celery Workers
                                      ├── worker-gpu (vlm_queue) → vLLM + vParse
                                      └── worker-cpu (default_queue) → PIL + piexif
```

### Extraction Pipeline

1. **PDF**: vParse OCR → text extraction (transcription → segments → legacy fallback) → content type detection → LLM extraction → ISBN validation → Open Library lookup (with retry) → confidence scoring
2. **Image**: PIL metadata extraction → EXIF parsing (camera, GPS, date, lens, DPI)
3. **Text/JSON**: Direct LLM semantic extraction (supports pre-OCR'd JSON with `transcription` field)

### Queue Routing

| File Type | Queue | Worker | LLM |
|-----------|-------|--------|-----|
| PDF | `vlm_queue` | worker-gpu | Yes |
| Text/JSON/MD | `vlm_queue` | worker-gpu | Yes |
| Images | `default_queue` | worker-cpu | No |

### Supported Models

| Model | Type | Min VRAM |
|-------|------|----------|
| Qwen/Qwen2.5-VL-7B-Instruct | Vision | 16 GB |
| Qwen/Qwen3-VL-30B-A3B-Instruct | Vision (MoE) | 6 GB |
| Qwen/Qwen3.5-9B | Text | 10 GB |
| google/gemma-3-27b-it | Text | 28 GB |
| google/gemma-4-31b-it | Text | 32 GB |
| google/gemma-4-E4B-it | Text | 8 GB |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_MODEL_ID` | HuggingFace model ID | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `VPARSE_API_URL` | vParse OCR API URL | `http://localhost:8000/file_parse` |
| `CELERY_BROKER_URL` | Redis broker URL | `redis://localhost:6379/0` |
| `CELERY_RESULT_BACKEND` | Redis result backend | `redis://localhost:6379/0` |
| `BOOKEXTRACTOR_UPLOAD_DIR` | Upload directory | `uploads` |
| `VLLM_DEVICE` | Target device: cuda, tpu, mps, cpu | Auto-detected |
| `VLLM_DTYPE` | Model precision | Auto-optimized |
| `VLLM_GPU_MEMORY_UTILIZATION` | GPU memory fraction (0.0-1.0) | `0.90` |
| `VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs (or `auto`) | Auto-optimized |
| `VLLM_MAX_MODEL_LEN` | Max context window for vLLM | `16384` |
| `MAX_BOOK_TEXT_CHARS` | Max chars fed to LLM for book extraction | `60000` |
| `MAX_MAGAZINE_TEXT_CHARS` | Max chars fed to LLM for magazine extraction | `15000` |

## Development

### Git Hooks

Configured via `.pre-commit-config.yaml`:
- **Commitizen**: Conventional commit messages
- **Ruff**: Linting and formatting
- **Bandit**: Security checks
- **Mypy**: Static type checking
- **Vulture**: Dead code detection
- **Pytest**: Full test suite with coverage (≥90%)

```bash
uv run pre-commit run --all-files
```

### CI/CD

GitLab CI pipeline (`.gitlab-ci.yml`):
1. **lint**: ruff, mypy, vulture
2. **test**: pytest with coverage report
3. **build**: Docker image (main branch only)

## License

MIT License. See `LICENSE` for details.
