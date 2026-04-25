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

`bookextractor` is optimized for Docker with a slim image and volume-mounted models.

### Run with Docker Compose (Recommended)

This method mounts your local models to the container to keep the image size small.

```bash
docker compose up --build -d
```

The API will be available at `http://localhost:8000`.

### Manual Docker Build

```bash
docker build -t bookextractor .
docker run -p 8000:8000 -v /path/to/model:/app/models/gemma.gguf bookextractor
```

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
