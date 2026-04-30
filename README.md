# BookExtractor 📚

A production-ready Python package for extracting structured metadata from scanned Telugu, Hindi, and English PDFs.

## 🚀 Overview

`bookextractor` is designed for high-accuracy metadata extraction from noisy, layout-agnostic scans. It uses a multi-stage pipeline combining adaptive region extraction, multilingual OCR, and a local Large Language Model (Gemma) to ensure validation-first results.

## ✨ Key Features

- **Adaptive Region Extraction**: Automatically identifies and crops high-signal areas (ISBN, Publisher info).
- **Multilingual Support**: Parallel OCR processing for English, Hindi, and Telugu.
- **Local LLM Integration**: Uses `gemma-4-E4B-it` (GGUF) for semantic field extraction without data leaving your machine.
- **Deterministic Validation**: Strict ISBN checksum verification; zero hallucinations for rigid fields.
- **Multi-Source Evidence**: Merges candidates across multiple pages and performs external API lookups (Open Library).

## 🛠 Installation

Requires **Python 3.10+**.

```bash
# Clone the repository
git clone <your-repo-url>
cd bookextractor

# Install dependencies and package
uv pip install -e .
```

## 📖 Usage

### CLI

On first run, BookExtractor automatically downloads `gemma-4-E4B-it-Q4_K_M.gguf` from `unsloth/gemma-4-E4B-it-GGUF` into `./models` if it is missing.

Projection (`MMPROJ`) is optional and only used/downloaded when you set projection env vars.

Extract metadata directly to a JSON file:
```bash
uv run bookextractor <input.pdf> <output.json>
```

Optional overrides:
```bash
# Use existing local files
export VLM_MODEL_PATH=/absolute/path/to/model.gguf
export MMPROJ_MODEL_PATH=/absolute/path/to/mmproj.gguf

# Or customize download URLs
export VLM_MODEL_URL=https://.../model.gguf
export MMPROJ_MODEL_URL=https://.../mmproj.gguf
```

Enable benchmark mode for debug info:
```bash
uv run bookextractor <input.pdf> <output.json> --benchmark
```

### API (FastAPI)

Start the web server:
```bash
uv run bookextractor --api
```
- **Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `GET /health`
- **Extract**: `POST /extract` (Multipart file upload)

## 🐳 Docker Support

The project runs as **two separate containers** orchestrated via Docker Compose:

| Service | Container Port | Host Port | Description |
|---------|---------------|-----------|-------------|
| `bookextractor` | 8000 | **8000** | BookExtractor FastAPI service |
| `vparse` | 8000 | **9000** | VParse (MinerU) OCR API service |

### Quick Start (Pipeline/Paddle – Default)

```bash
docker compose up --build -d
```

This builds and starts both services. The VParse container installs **only pipeline dependencies** (`torch`, `torchvision`, `onnxruntime` — no `vllm`, no CUDA libs).

### Other OCR Backends

Each mode has its own Dockerfile with **only the dependencies it needs**:

```bash
# Pipeline/Paddle (default) – multi-model, multilingual
docker compose up --build -d

# Lite – Tesseract-only, smallest image, no torch
VPARSE_DOCKERFILE=Dockerfile.vparse.lite docker compose up --build -d

# Hybrid – pipeline + VLM combined
VPARSE_DOCKERFILE=Dockerfile.vparse.hybrid docker compose up --build -d

# VLM – vision-language model (GPU recommended)
VPARSE_DOCKERFILE=Dockerfile.vparse.vlm docker compose up --build -d
```

| Mode | Dockerfile | Pip extras | Image size |
|------|-----------|------------|------------|
| **Pipeline/Paddle** | `Dockerfile.vparse` | `.[pipeline,api]` | ~2 GB |
| **Lite** | `Dockerfile.vparse.lite` | `.[lite,api]` | ~500 MB |
| **Hybrid** | `Dockerfile.vparse.hybrid` | `.[pipeline,vlm,api]` | ~3 GB |
| **VLM** | `Dockerfile.vparse.vlm` | `.[vlm,api]` | ~2.5 GB |

### Model Persistence

Models are stored in a Docker named volume `models`. They are downloaded on first run and reused across container restarts — no re-download on rebuild.

## 🏗 Architecture

1.  **Smart Selection**: Analyzes the first 7 pages (where metadata usually lives).
2.  **Adaptive Cropping**: Searches for keywords (ISBN, Edition) to create high-probability crops.
3.  **OCR Pass**: Dual-pass EasyOCR (English+Telugu, English+Hindi).
4.  **ISBN Validation**: Regex + Checksum (Modulo 10/11).
5.  **Semantic Extraction**: Gemma LLM processes OCR text for Title, Author, and Publisher.
6.  **External Verification**: Valid ISBNs are checked against Open Library API.
7.  **Merge & Score**: Field-wise merging with confidence scoring.

## 🤝 GitLab / CI Standards

This project follows professional Python standards:
- **Linting**: Complies with `ruff` standards.
- **Packaging**: Uses `pyproject.toml` (PEP 621).
- **Type Safety**: Pydantic v2 models for all data structures.
- **Environment**: Optimized for `uv` or `pip`.

## 📄 License

MIT License. See `LICENSE` for details.
