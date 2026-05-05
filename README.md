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

# Set up Git hooks (Mandatory for contributors)
uv run pre-commit install
```

## 🛠 Development

This project uses the standard `pre-commit` framework for validation.

### Git Hooks

The following hooks are configured via `.pre-commit-config.yaml`:
- **Commitizen**: Enforces conventional commit messages.
- **Ruff**: Linting and formatting.
- **Bandit**: Security checks.
- **Mypy**: Static type checking.
- **Vulture**: Dead code detection.
- **Pytest**: Full test suite with coverage enforcement.

To manually install or refresh the hooks:
```bash
uv run pre-commit install
```

### Manual Validation

You can run the full suite of hooks manually at any time:
```bash
uv run pre-commit run --all-files
```

## 📖 Usage

### CLI

The CLI supports both the legacy shortcut form and explicit subcommands.

Extract metadata directly to a JSON file:

```bash
uv run bookextractor input.pdf output.json
uv run bookextractor extract input.pdf output.json
```

Use a specific OCR language for PDFs:

```bash
uv run bookextractor extract input.pdf output.json --lang te
uv run bookextractor extract input.pdf output.json --lang devanagari
```

Enable benchmark mode for debug info:

```bash
uv run bookextractor extract input.pdf output.json --benchmark
```

### API (FastAPI)

Start the web server:

```bash
uv run bookextractor --api
uv run bookextractor api
uv run bookextractor api --host 0.0.0.0 --port 8000
```

- **Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `GET /health`
- **Extract**: `POST /extract` (Multipart file upload)

## 🐳 Docker Support

The project runs using Docker Compose. By default, only **BookExtractor** and its **model-downloader** are started. The VParse (MinerU) OCR API service is optional and is pulled directly from the official upstream repository on demand via Docker Compose profiles.

### Quick Start (Metadata Indexing Only)

To start only the BookExtractor service and download the Gemma LLM:

```bash
docker compose up --build -d
```

This builds the BookExtractor container and starts the `model-downloader` to fetch the Gemma model into a shared volume.

### Running with VParse OCR Backends

If you need the OCR capabilities, you can include VParse by specifying a **Docker profile**. When a VParse profile is specified, Docker Compose automatically clones the MinerU repository from Git and builds the required mode without needing local Dockerfiles.

Available profiles for VParse:

- `pipeline`: Multi-model, multilingual OCR using Paddle (CPU/GPU)
- `vlm`: Vision-language model for OCR (GPU recommended)
- `hybrid`: Combined pipeline + VLM

**Example: Start BookExtractor with VParse Pipeline Mode**

```bash
docker compose --profile pipeline up --build -d
```

| Service         | Container Port | Host Port | Description                     |
| --------------- | -------------- | --------- | ------------------------------- |
| `bookextractor` | 8000           | **8000**  | BookExtractor FastAPI service   |
| `vparse`        | 8000           | **9000**  | VParse (MinerU) OCR API service |

### Model Persistence

Models (both Gemma LLM and VParse OCR weights) are stored in a Docker named volume `models`. They are downloaded by the `model-downloader` service on first run and reused across container restarts — no re-download on rebuild. You can monitor the download progress by checking the logs:

```bash
docker compose logs -f model-downloader
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
