# BookExtractor — Architecture Document

**Branch:** `feat/celery-redis-integration` (7 commits ahead of `develop`)
**Date:** 2026-05-13
**Status:** Ready for merge review

---

## 1. System Overview

BookExtractor extracts structured metadata from scanned Telugu/Hindi/English books, magazines, and images using a multi-pipeline architecture combining OCR (vParse), LLM inference (vLLM), and rule-based validation.

### Core Capabilities
- **PDF extraction** — OCR via vParse API → LLM metadata extraction
- **Image extraction** — EXIF/metadata parsing (PIL + piexif)
- **Text/JSON extraction** — Direct LLM semantic extraction from plain text or pre-OCR'd JSON
- **Magazine detection** — Auto-detects magazine vs book content, uses different prompts
- **ISBN validation** — Regex extraction + checksum validation + Open Library API lookup
- **Hardware optimization** — Auto-detects GPU/CPU/TPU/MPS and configures vLLM accordingly

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐   │
│  │   CLI (Typer) │  │  FastAPI     │  │  Celery Worker (CLI command)    │   │
│  │  bookextractor│  │  POST /extract│  │  bookextractor worker -q ...   │   │
│  │  extract      │  │  POST /extract/async                              │   │
│  │  model list   │  │  GET  /jobs/{id}                                  │   │
│  │  model download│ │  GET  /health                                     │   │
│  └───────┬───────┘  └──────┬───────┘  └──────────────┬─────────────────┘   │
│          │                 │                          │                      │
└──────────┼─────────────────┼──────────────────────────┼──────────────────────┘
           │                 │                          │
           ▼                 ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        ExtractionPipeline                             │   │
│  │                                                                       │   │
│  │  process_pdf() ──► vParse OCR ──► extract_from_text()                │   │
│  │  process_image() ──► extract_image_metadata()                        │   │
│  │  process_text_file() ──► extract_from_text()                         │   │
│  │                                                                       │   │
│  │  extract_from_text() ──┬─► _extract_book_metadata()                  │   │
│  │                        └─► _extract_magazine_metadata()              │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             │                                                │
│              ┌──────────────┼──────────────┐                                │
│              ▼              ▼              ▼                                │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────────┐                       │
│  │  VLMClient    │ │  validation  │ │ external_api │                       │
│  │  (vLLM LLM)   │ │  (ISBN)      │ │ (OpenLibrary)│                       │
│  │  Singleton    │ │  Regex+Check │ │  HTTP lookup │                       │
│  └───────────────┘ └──────────────┘ └──────────────┘                       │
│                                                                              │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐    │
│  │ hardware.py   │ │ models.py    │ │ models_reg.  │ │ vparse_client │    │
│  │ GPU detection │ │ Pydantic     │ │ Cache scanner│ │ HTTP to vParse│    │
│  │ vLLM config   │ │ Data models  │ │ HF_HOME aware│ │ OCR API       │    │
│  └───────────────┘ └──────────────┘ └──────────────┘ └───────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
           │                                                  │
           ▼                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ASYNC / QUEUE LAYER                                 │
│                                                                              │
│  ┌─────────────────┐          ┌─────────────────┐                           │
│  │  Redis Broker   │◄────────►│  Celery Workers  │                           │
│  │  redis:6379/0   │          │                  │                           │
│  │  (task queue)   │          │  worker-gpu      │                           │
│  │  (result backend)│         │  - vlm_queue     │                           │
│  └─────────────────┘          │  - concurrency 2 │                           │
│                               │  - NVIDIA GPU    │                           │
│                               │                  │                           │
│                               │  worker-cpu      │                           │
│                               │  - default_queue │                           │
│                               │  - concurrency 8 │                           │
│                               │  - CPU only      │                           │
│                               └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐  │
│  │  vParse API  │  │  HuggingFace │  │  Open Library API                │  │
│  │  (mineru)    │  │  Model Cache │  │  (ISBN validation)               │  │
│  │  OCR engine  │  │  ~/.cache/   │  │  https://openlibrary.org         │  │
│  │  port 8000   │  │  huggingface │  │                                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Breakdown

### 3.1 `main.py` — Entry Points
| Component | Purpose |
|-----------|---------|
| `cli_app` (Typer) | CLI commands: `extract`, `api`, `worker`, `hardware-info`, `model` |
| `app` (FastAPI) | REST API: `/health`, `/extract` (sync), `/extract/async`, `/jobs/{id}` |
| `ALLOWED_EXTENSIONS` | `.pdf`, `.md`, `.json`, `.txt`, `.jpg`, `.jpeg`, `.png`, `.webp`, `.tiff`, `.tif` |
| `extract_async` | Saves file → routes to correct Celery task → returns `job_id` |
| `get_job_status` | Polls `celery_app.AsyncResult(job_id)` for status/result |
| `model download` | Interactive model picker using `questionary.checkbox` + `snapshot_download` |

### 3.2 `pipeline.py` — Core Extraction Logic
| Method | Flow | LLM Required |
|--------|------|--------------|
| `process_pdf()` | vParse OCR → text cleaning → `extract_from_text()` | Yes |
| `process_image()` | PIL open → EXIF extraction → `ImageMetadata` | No |
| `process_text_file()` | Read file → detect JSON with transcription → `extract_from_text()` | Yes |
| `extract_from_text()` | Detect content type → route to book or magazine extraction | Yes |
| `_extract_book_metadata()` | ISBN regex → LLM extraction → Open Library lookup → confidence scoring | Yes |
| `_extract_magazine_metadata()` | LLM magazine prompt → confidence scoring | Yes |

### 3.3 `tasks.py` — Celery Task Definitions
| Task | Queue | LLM | Concurrency Target |
|------|-------|-----|-------------------|
| `extract_pdf` | `vlm_queue` | Yes (`load_llm=True`) | 2 (GPU) |
| `extract_image` | `default_queue` | No (`load_llm=False`) | 8 (CPU) |
| `extract_text` | `vlm_queue` | Yes (`load_llm=True`) | 2 (GPU) |

Pipeline instances are lazy-loaded singletons per worker process.

### 3.4 `celery_config.py` — Queue Configuration
```
broker_url = redis://localhost:6379/0
result_backend = redis://localhost:6379/0

task_routes:
  extract_image → default_queue  (CPU-only, EXIF parsing)
  extract_pdf   → vlm_queue       (GPU, LLM inference)
  extract_text  → vlm_queue       (GPU, LLM inference)

worker_prefetch_multiplier = 1    # One task at a time per worker
worker_max_tasks_per_child = 100  # Restart after 100 tasks (memory leak prevention)
task_time_limit = 3600            # 1 hour hard limit
```

### 3.5 `vlm_client.py` — LLM Wrapper
- **Singleton pattern** — `VLMClient.get_instance()` ensures one vLLM engine per process
- **Auto-detects cached model** from HF cache if no model ID provided
- **Hardware-optimized** via `hardware.get_vllm_config()`
- Supports Gemma 4, Qwen2.5-VL, Qwen3-VL, Qwen3.5

### 3.6 `hardware.py` — Hardware Detection
Detection priority: TPU → NVIDIA (torch) → NVIDIA (NVML) → Apple MPS → CPU

| GPU | dtype | memory_util | tensor_parallel |
|-----|-------|-------------|-----------------|
| T4 | float16 | 0.70 | 1 |
| V100 | bfloat16 | 0.85 | 1 |
| A100 | bfloat16 | 0.90 | count |
| RTX 3090/4090 | bfloat16 | 0.85 | 1 |
| <8GB | float16 | 0.60 | 1 |

Env var overrides: `VLLM_DEVICE`, `VLLM_DTYPE`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_TENSOR_PARALLEL_SIZE`

### 3.7 `models_registry.py` — Model Cache Management
- **HF_HOME aware** — respects `$HF_HOME` env var (fixed from hardcoded path)
- Scans `~/.cache/huggingface/hub` for cached models
- 6 registered models with VRAM requirements
- CLI commands: `model list`, `model download`, `model remove`, `model cache`

### 3.8 `vparse_client.py` — vParse OCR Client
- HTTP POST to vParse API with PDF file + language parameter
- Returns `content_list` (preferred) or `md_content`
- Timeout: 900s (15 min for large PDFs)

### 3.9 `models.py` — Pydantic Data Models
- `BookMetadata` — title, author, publisher, isbn, published_date, confidence
- `MagazineMetadata` — magazine_name, editor, publisher, issue_date, issue_number, price
- `ImageMetadata` — dimensions, format, EXIF (camera, GPS, date, lens, DPI)
- `BenchmarkResult` — wraps `ExtractionResult` + debug info

### 3.10 `validation.py` — ISBN Validation
- Regex extraction of ISBN-10 and ISBN-13 candidates
- Checksum validation (ISBN-10 mod 11, ISBN-13 mod 10)

### 3.11 `external_api.py` — Open Library Lookup
- Async HTTP GET to `openlibrary.org/api/books`
- Enriches LLM-extracted metadata with authoritative data

---

## 4. Docker Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        docker-compose.yml                            │
│                                                                       │
│  ┌─────────────┐    ┌──────────┐    ┌──────────────┐                │
│  │ bookextractor│    │  redis   │    │model-downloader│              │
│  │ (FastAPI)   │    │  6379    │    │ (HF download) │                │
│  │ GPU + LLM   │    │          │    │               │                │
│  └──────┬──────┘    └────┬─────┘    └───────┬───────┘                │
│         │                │                   │                         │
│  ┌──────┴──────┐         │                   │                         │
│  │ worker-gpu  │         │                   │                         │
│  │ vlm_queue   │◄────────┘                   │                         │
│  │ concurrency 2│                            │                         │
│  └─────────────┘                             │                         │
│  ┌─────────────┐                             │                         │
│  │ worker-cpu  │                             │                         │
│  │ default_queue│                            │                         │
│  │ concurrency 8│                            │                         │
│  └─────────────┘                             │                         │
│                                               │                         │
│  ┌─────────────────────────────────────────────┘                         │
│  │ (VParse services - optional profiles)                                 │
│  │  vparse-pipeline  [profile: pipeline]                                 │
│  │  vparse-vlm       [profile: vlm]                                      │
│  │  vparse-hybrid    [profile: hybrid]                                   │
│  └──────────────────────────────────────────────────────────────────────┘
│
│  Volumes:
│    ~/.cache/huggingface:/models/huggingface  (bind mount - shared cache)
│    models:/models                            (named volume)
│    ./uploads:/app/uploads                    (persistent uploads)
│
│  Env:
│    VLLM_MODEL=${VLLM_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}
│    HF_HOME=/models/huggingface
│    CELERY_BROKER_URL=redis://redis:6379/0
│    CELERY_RESULT_BACKEND=redis://redis:6379/0
└──────────────────────────────────────────────────────────────────────┘
```

### Volume Mount Strategy
All services mount `~/.cache/huggingface:/models/huggingface` so models downloaded by `model-downloader` are visible to `bookextractor`, `worker-gpu`, `worker-cpu`, and all vParse services.

---

## 5. Data Flow

### 5.1 PDF Extraction (Async)
```
Client → POST /extract/async
  → Save file to uploads/{uuid}_{filename}
  → extract_pdf_task.delay(path, lang)
  → Redis broker queues to vlm_queue
  → worker-gpu picks up task
    → get_pipeline(load_llm=True)  [lazy singleton]
    → parse_pdf_via_vparse(path, lang)  [HTTP to vParse]
    → extract_from_text(text, lang)
      → _detect_content_type(text) → book or magazine
      → _extract_book_metadata() or _extract_magazine_metadata()
        → extract_semantic_fields() / extract_magazine_semantic_fields()  [vLLM]
        → extract_isbn_candidates() [regex]
        → lookup_isbn() [Open Library API]
        → calculate_confidence()
    → Return JSON result
  → Redis stores result
  → Client polls GET /jobs/{id}
  → Cleanup: delete uploaded file
```

### 5.2 Image Extraction (Async)
```
Client → POST /extract/async
  → extract_image_task.delay(path)
  → Redis broker queues to default_queue
  → worker-cpu picks up task
    → get_pipeline(load_llm=False)  [no LLM loaded]
    → process_image(path)
      → extract_image_metadata(path)  [PIL + piexif]
    → Return JSON result
  → Cleanup: delete uploaded file
```

### 5.3 CLI Extraction (Sync)
```
bookextractor extract input.pdf output.json --lang te
  → _extract_file()
  → get_pipeline()  [singleton in process]
  → process_pdf() / process_text_file() / process_image()
  → Write JSON to output path
```

---

## 6. Merge Readiness Assessment

### 6.1 CI Status
| Check | Status |
|-------|--------|
| ruff check | ✅ Pass |
| ruff format | ✅ Pass |
| mypy | ✅ Pass |
| vulture | ✅ Pass |
| pytest (140 tests) | ✅ Pass, 95% coverage |
| pre-commit hooks | ✅ Pass |

### 6.2 Commits Ahead of develop
| Commit | Type | Description |
|--------|------|-------------|
| `1e7664c` | fix | Improve JSON parsing, increase LLM context window |
| `4098b82` | test | Fixed linting issues |
| `f191e77` | fix | Modify CI dependencies to fix pipeline |
| `2f05deb` | feat | Implement Celery and Redis background task queue |
| `d1283d6` | fix | Reconcile merge conflicts, repair stale mocks, patch magazine error-handling |
| `844cb08` | feat | Dynamic model selection and shared Hugging Face cache |
| `f93a29f` | fix | Remove deprecated resume_download from snapshot_download |

### 6.3 Known Issues / Risks
| Issue | Severity | Status |
|-------|----------|--------|
| `vllm` imported at module level in `pipeline.py` | Medium | CPU workers load ~200-300MB extra per process. Lazy import would reduce to ~100MB/process |
| `Dockerfile.bookextractor` uses CUDA 12.4 but docker-compose sets `HF_HOME=/root/.cache/huggingface/hub` (different from compose bind mount) | Low | Runtime env var `HF_HOME=/models/huggingface` overrides Dockerfile default |
| `worker-cpu` has `HF_HOME` set but doesn't need it (no LLM) | Cosmetic | No functional impact |
| `pyproject.toml` has duplicate `pytest-httpx>=0.36.2` entry | Cosmetic | uv handles dedup, no functional impact |
| `_extract_file` uses `os.path.dirname(os.path.abspath(output_json))` which fails if output is just a filename | Low | Works for paths with directories, fails for bare filenames |

### 6.4 Recommendation
**Ready to merge.** All CI checks pass, 140/140 tests pass at 95% coverage, no blocking issues. The known issues are cosmetic or low-severity optimizations that can be addressed in follow-up PRs.

---

## 7. File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `bookextractor/main.py` | 383 | CLI + FastAPI entry points |
| `bookextractor/pipeline.py` | 320 | Core extraction pipeline |
| `bookextractor/celery_config.py` | 38 | Celery broker/queue config |
| `bookextractor/tasks.py` | 64 | Celery task definitions |
| `bookextractor/vlm_client.py` | 106 | vLLM singleton wrapper |
| `bookextractor/hardware.py` | 215 | GPU/CPU/TPU detection + vLLM config |
| `bookextractor/models_registry.py` | 120 | Model cache scanner + registry |
| `bookextractor/models.py` | 74 | Pydantic data models |
| `bookextractor/vparse_client.py` | 35 | vParse OCR HTTP client |
| `bookextractor/external_api.py` | 29 | Open Library ISBN lookup |
| `bookextractor/validation.py` | 39 | ISBN regex + checksum |
| `bookextractor/image_utils.py` | 109 | Image metadata + EXIF extraction |
| `scripts/setup_models.py` | 44 | Docker model downloader |
| `docker-compose.yml` | 238 | Service orchestration |
| `Dockerfile.bookextractor` | 22 | BookExtractor container |
| `Dockerfile.model-downloader` | — | Model download container |
| `tests/` | — | 140 tests, 95% coverage |
