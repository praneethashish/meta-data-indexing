# BookExtractor — CLI Mode Architecture

**Branch:** `feat/anyllm-integration`
**Date:** 2026-05-22

---

## 1. System Overview

BookExtractor extracts structured metadata from scanned Telugu/Hindi/English books, magazines, and images. In CLI mode, everything runs in a single process — no Redis, no Celery, no network services (except external APIs).

### Core Capabilities

| Capability | Backend | GPU Required |
|-----------|---------|-------------|
| PDF text extraction | anyLLm (remote HTTP) **or** vLLM (local) | Only if local vLLM |
| JSON/text extraction | anyLLm (remote HTTP) **or** vLLM (local) | Only if local vLLM |
| Image EXIF metadata | PIL + piexif | No |
| Image VLM analysis | vLLM (local only) | Yes |
| Magazine detection | Rule-based pattern matching | No |
| ISBN validation | Regex + checksum + Open Library API | No |

### LLM Backend Decision

```
--backend flag (CLI) or llm_backend param (programmatic)
    ├── "local"  → LocalVLLMClient (ignores remote env vars)
    ├── "remote" → AnyLLMClient (requires BOOKEXTRACTOR_LLM_* vars)
    └── "auto"   → env-driven (default):
                   .env has all 3 BOOKEXTRACTOR_LLM_* vars?
                       ├── Yes → AnyLLMClient (remote HTTP, no GPU)
                       └── No  → LocalVLLMClient (local vLLM on GPU)
```

---

## 2. Architecture Diagram

### 2.1 CLI Mode — Overall System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SINGLE PROCESS                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CLI (Typer)                                                         │   │
│  │  bookextractor extract <input> <output.json> [options]               │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             │                                                │
│                             ▼                                                │
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
│  │  llm_client   │ │  validation  │ │ external_api │                       │
│  │  (lazy init)  │ │  (ISBN)      │ │ (OpenLibrary)│                       │
│  │               │ │  Regex+Check │ │  HTTP lookup │                       │
│  └───────┬───────┘ └──────────────┘ └──────────────┘                       │
│          │                                                                  │
│  ┌───────┴──────────────────────────────────────────────────────────┐     │
│  │  LLM Backend Selection (create_llm_client)                        │     │
│  │                                                                    │     │
│  │  .env has BOOKEXTRACTOR_LLM_*?                                     │     │
│  │  ├── Yes → AnyLLMClient ──► anyllm ──► HTTP POST to remote URL    │     │
│  │  └── No  → LocalVLLMClient ──► ModelClient ──► ModelManager       │     │
│  │                              └─► vLLM engine (GPU)                │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐    │
│  │ vision_client │ │ hardware.py  │ │ models.py    │ │ vparse_client │    │
│  │ (ModelClient) │ │ GPU detection│ │ Pydantic     │ │ HTTP to vParse│    │
│  │ (GPU, vLLM)   │ │ vLLM config  │ │ Data models  │ │ OCR API       │    │
│  └───────────────┘ └──────────────┘ └──────────────┘ └───────────────┘    │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────────┐                     │
│  │ image_utils   │ │ content_det. │ │confidence_sc.│                     │
│  │ PIL + piexif  │ │ pattern match│ │ scoring logic│                     │
│  └───────────────┘ └──────────────┘ └──────────────┘                     │
│  ┌───────────────┐ ┌──────────────┐                                     │
│  │ config.py     │ │ prompts.py   │                                     │
│  │ Settings      │ │ LLM prompts  │                                     │
│  └───────────────┘ └──────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘
            │                                                  │
            ▼                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐  │
│  │  vParse API  │  │  HuggingFace │  │  Remote LLM (anyLLM)             │  │
│  │  (mineru)    │  │  Model Cache │  │  (Ollama, OpenAI, etc.)          │  │
│  │  OCR engine  │  │  ~/.cache/   │  │  HTTP POST /v1/chat/completions  │  │
│  │  port 8000   │  │  huggingface │  │                                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────┐                                       │
│  │  Open Library API                │                                       │
│  │  (ISBN validation)               │                                       │
│  │  https://openlibrary.org         │                                       │
│  └──────────────────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 LLM Client Selection Flow

```
                    extract_semantic_fields(text)
                              │
                              ▼
                    ensure_llm_client()
                              │
                              ▼
                    create_llm_client(backend=...)
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                 "local"   "remote"   "auto" (default)
                    │         │         │
                    ▼         │         ▼
            LocalVLLMClient   │   has_partial_config?
                    │         │         │
                    │         ▼         ├── Yes → RuntimeError
                    │   RuntimeError    │          "Incomplete config"
                    │   if partial      │
                    │   config          │   has_remote_config?
                    │         │         ├── Yes → AnyLLMClient
                    │         ▼         │
                    │   AnyLLMClient    │   └── No  → LocalVLLMClient
                    │         │
                    ▼         ▼
            ModelClient   _prefixed_model = "openai/{model}"
            .get()        anyllm.chat(model=prefixed_model)
                    │         │
                    ▼         ▼
            vLLM engine   HTTP POST to remote URL
            (GPU)         parse JSON response
                    │         │
                    ▼         ▼
            generate()    return dict[str, Any]
```

### 2.3 File Processing Flow

```
                    bookextractor extract input output
                              │
                              ▼
                    _extract_file()
                              │
                    ┌─────────┼─────────┬────────────┐
                    │         │         │            │
                    ▼         ▼         ▼            ▼
                  .pdf      .json     .jpg/.png   .md/.txt
                    │       / .md       / .webp
                    │       / .txt      / .tiff
                    ▼         │            │
              ExtractionPipeline           │
              load_vlm=True                │
                    │                      │
          ┌─────────┴──────────┐           │
          │                    │           ▼
          ▼                    ▼    ExtractionPipeline
    process_pdf()      process_text_file()  load_vlm=False
          │                    │            │
          ▼                    ▼            ▼
    vParse OCR          Parse JSON/     extract_image_
    → full_text         Read text       metadata()
          │                    │            │
          ▼                    ▼            ▼
    extract_from_text   extract_from_text  If vision_client:
          │                    │            describe_image()
          ▼                    ▼            │
    detect_content_type       │            ▼
          │                    │       EXIF + VLM data
          ▼                    ▼
    Book or Magazine?    Book or Magazine?
          │                    │
          ▼                    ▼
    LLM extraction       LLM extraction
    ISBN lookup          ISBN lookup
    Confidence scoring   Confidence scoring
          │                    │
          └────────┬───────────┘
                   │
                   ▼
          json.dump(output.json)
          ensure_ascii=False
```

---

## 3. Entry Point

### CLI Command

```bash
uv run bookextractor extract input.json output.json
uv run bookextractor extract input.pdf output.json --lang te
uv run bookextractor extract input.jpg output.json --vlm
```

### File: `bookextractor/__init__.py`

```
1. dotenv.load_dotenv(".env") — auto-loads env vars
2. Imports main.py (app, cli_app)
```

### File: `bookextractor/main.py`

```
cli_app (Typer)
    └── @cli_app.command("extract") → extract_command()
        └── asyncio.run(_extract_file(...))
```

---

## 4. CLI Data Flow by File Extension

### 4.1 `.pdf` Files

```
bookextractor extract book.pdf output.json
    │
    ▼ main.py: extract_command()
    │   load_vlm = (not is_image) or use_vlm → True
    │
    ▼ main.py: _extract_file()
    │   p = ExtractionPipeline(load_vlm=True)
    │
    ▼ pipeline.py: ExtractionPipeline.__init__()
    │   ├── llm_client = None (lazy)
    │   ├── vision_client = ModelClient.get_instance()
    │   │   └── Tries local vLLM → OOM? → caught → None (graceful skip)
    │   └── llm_backend = backend.value (auto|remote|local)
    │
    ▼ pipeline.py: process_pdf()
    │   1. parse_pdf_via_vparse(pdf_path, lang="en")
    │      └── HTTP POST to VPARSE_API_URL with PDF file
    │      └── Returns content_list[] or md_content
    │   2. Join content_list[].text → full_text
    │   3. Clean: remove image refs, collapse whitespace
    │   4. extract_from_text(full_text)
    │
    ▼ pipeline.py: extract_from_text()
    │   detect_content_type(text) → "book" or "magazine"
    │
    ▼ pipeline.py: _extract_book_metadata() or _extract_magazine_metadata()
    │   1. extract_semantic_fields(text)
    │      └── ensure_llm_client()
    │          └── create_llm_client(backend=self._llm_backend)
    │              ├── backend="local" → LocalVLLMClient
    │              ├── backend="remote" → AnyLLMClient (requires env vars)
    │              └── backend="auto" → env-driven (has_remote? → AnyLLMClient : LocalVLLMClient)
    │      └── LLM returns JSON dict
    │   2. extract_isbn_candidates(text) → regex scan
    │   3. lookup_isbn(isbn) → HTTP to openlibrary.org
    │   4. calculate_book_confidence() or calculate_magazine_confidence()
    │
    ▼ main.py: json.dump(result, f, indent=2, ensure_ascii=False)
    │
    ▼ output.json written
```

### 4.2 `.json` Files (Pre-OCR'd)

```
bookextractor extract ocrd.json output.json
    │
    ▼ main.py: extract_command()
    │   load_vlm = (not is_image) or use_vlm → True
    │
    ▼ pipeline.py: ExtractionPipeline.__init__()
    │   ├── llm_client = None (lazy)
    │   ├── vision_client = ModelClient.get_instance()
    │   │   └── Tries local vLLM → OOM? → caught → None
    │   └── llm_backend = backend.value (auto|remote|local)
    │
    ▼ pipeline.py: process_text_file()
    │   1. Read JSON file → json.loads()
    │   2. Check "transcription" key → if found, use it
    │   3. Check "segments" or "content_list" → sort by reading_order → join .text
    │   4. Fallback: json.dumps(entire JSON)
    │   5. extract_from_text(clean_text)
    │
    ▼ pipeline.py: extract_from_text()
    │   detect_content_type(text) → "book" or "magazine"
    │   └── Same LLM flow as PDF (see 3.1)
    │
    ▼ output.json written
```

### 4.3 `.md` / `.txt` Files

```
bookextractor extract book.md output.json
    │
    ▼ pipeline.py: process_text_file()
    │   1. Read file content as plain text
    │   2. extract_from_text(content)
    │   └── Same LLM flow as PDF (see 3.1)
    │
    ▼ output.json written
```

### 4.4 `.jpg` / `.jpeg` / `.png` / `.webp` / `.tiff` / `.tif` Files

```
bookextractor extract photo.jpg output.json
    │
    ▼ main.py: extract_command()
    │   load_vlm = (not is_image) or use_vlm → False (default)
    │   load_vlm = True if --vlm flag passed
    │
    ▼ pipeline.py: ExtractionPipeline.__init__()
    │   ├── llm_client = None (lazy, not used for images)
    │   ├── vision_client = ModelClient.get_instance() if load_vlm=True
    │   │   └── Tries local vLLM → OOM? → caught → None
    │   └── llm_backend = backend.value (auto|remote|local)
    │
    ▼ pipeline.py: process_image()
    │   1. extract_image_metadata(image_path)
    │      └── PIL open → dimensions, format, color_space
    │      └── piexif → EXIF (camera, GPS, date, lens, DPI)
    │   2. If vision_client is not None:
    │      └── vision_client.describe_image() → VLM analysis
    │      └── Returns: description, text_content, language, scene_classification, entities
    │   3. Build ExtractionResult(image_metadata, image_vlm_metadata)
    │
    ▼ output.json written
```

---

## 5. Module Breakdown

### 5.1 `__init__.py` — Package Bootstrap

| Responsibility | Detail |
|---|---|
| `.env` auto-load | `dotenv.load_dotenv()` before any module reads env vars |
| Exports | `app` (FastAPI), `cli_app` (Typer), `__version__` |

### 5.2 `main.py` — CLI Entry Points

| Component | Purpose |
|---|---|
| `LLMBackend` | Enum: `auto`, `remote`, `local` — controls LLM backend selection |
| `extract_command` | Parses CLI args (incl. `--backend`/`-b`) → calls `_extract_file()` |
| `_extract_file` | Validates backend mode, file type routing → pipeline creation → JSON output |
| `api_command` | Starts FastAPI server via uvicorn |
| `worker_command` | Starts Celery worker subprocess |
| `hardware_info_command` | Displays GPU detection + vLLM config suggestions |
| `model_app` | Sub-commands: `list`, `download`, `remove`, `cache` |

### 5.3 `pipeline.py` — Core Extraction Logic

| Method | Flow | LLM Required |
|---|---|---|
| `process_pdf()` | vParse OCR → text cleaning → `extract_from_text()` | Yes |
| `process_image()` | EXIF extraction → optional VLM analysis | No (VLM optional) |
| `process_text_file()` | Read → detect JSON structure (`segments`/`content_list`/`transcription`) → extract text → `extract_from_text()` | Yes |
| `extract_from_text()` | Detect content type → route to book or magazine | Yes |
| `_extract_book_metadata()` | LLM extraction → ISBN → Open Library → confidence | Yes |
| `_extract_magazine_metadata()` | LLM magazine prompt → confidence | Yes |
| `ensure_llm_client()` | Lazy init: `create_llm_client(backend=self._llm_backend)` → AnyLLMClient or LocalVLLMClient | — |

### 5.4 `llm_clients.py` — LLM Client Abstraction

| Class/Function | Purpose | Backend |
|---|---|---|
| `BaseLLMClient` | Abstract interface | — |
| `AnyLLMClient` | Remote LLM via HTTP | `anyllm` package → OpenAI-compatible endpoint (model prefixed as `provider/model`) |
| `LocalVLLMClient` | Local vLLM engine | `ModelClient` → `ModelManager` → vLLM |
| `create_llm_client(backend)` | Factory — accepts `auto`, `remote`, `local` | Returns appropriate client based on mode + env vars |
| `has_remote_llm_config()` | Checks all 3 BOOKEXTRACTOR_LLM_* vars | — |
| `has_partial_remote_llm_config()` | Detects incomplete config (error case) | — |

**AnyLLMClient Model Prefix:** anyLLM's `parse_model_string()` extracts the provider from the first `/` in the model string. Since HF model IDs like `Qwen/Qwen2.5-7B-Instruct` contain `/`, the client prefixes the model as `openai/Qwen/Qwen2.5-7B-Instruct` so anyLLM correctly identifies `openai` as the provider.

### 5.5 `model_client.py` — vLLM Engine Client

| Method | Purpose |
|---|---|
| `get_instance()` | Singleton — one vLLM engine per process |
| `generate_and_extract()` | Text generation → JSON parsing |
| `describe_image()` | Vision analysis — multimodal image understanding |
| `is_ready()` | Check if engine is loaded |

### 5.6 `model_manager.py` — vLLM Lifecycle Manager

| Method | Purpose |
|---|---|
| `get_instance()` | Thread-safe singleton |
| `get_or_create()` | Initialize vLLM engine with hardware-optimized config |
| `get_model()` | Return loaded model |
| `is_loaded()` | Check if model is ready |
| `reset()` | Reset singleton (testing only) |

### 5.7 `config.py` — Centralized Configuration

| Category | Env Vars |
|---|---|
| Remote LLM (anyLLM) | `BOOKEXTRACTOR_LLM_MODEL`, `BASE_URL`, `API_KEY`, `PROVIDER` (default: `openai`) |
| Local vLLM | `VLLM_MODEL_ID`, `VLLM_MODEL`, `VLLM_DEVICE`, `VLLM_DTYPE`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_TENSOR_PARALLEL_SIZE` |
| API | `API_HOST`, `API_PORT` |
| Celery | `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, etc. |
| External Services | `VPARSE_API_URL`, `VPARSE_TIMEOUT`, `OPENLIBRARY_TIMEOUT` |
| File Types | `ALLOWED_EXTENSIONS`, `IMAGE_EXTENSIONS` (class-level, not env-overridable) |

**Note:** `Settings` uses `__init__` for testability — env vars are read at instantiation time, not at class definition.

### 5.8 `vparse_client.py` — vParse OCR Client

| Method | Detail |
|---|---|
| `parse_pdf_via_vparse()` | HTTP POST PDF to vParse API, returns OCR text |
| Timeout | 900s (15 min for large PDFs) |

### 5.9 `external_api.py` — Open Library Lookup

| Method | Detail |
|---|---|
| `lookup_isbn()` | Async HTTP GET to openlibrary.org, enriches metadata |

### 5.10 `validation.py` — ISBN Validation

| Function | Detail |
|---|---|
| `extract_isbn_candidates()` | Regex scan for ISBN-10 and ISBN-13 patterns |

### 5.11 `image_utils.py` — Image Metadata Extraction

| Function | Detail |
|---|---|
| `extract_image_metadata()` | PIL + piexif → dimensions, format, EXIF, GPS, DPI |

### 5.12 `content_detector.py` — Content Type Detection

| Function | Detail |
|---|---|
| `detect_content_type()` | Scans text for magazine patterns → returns "book" or "magazine" |

### 5.13 `confidence_scorer.py` — Confidence Calculation

| Function | Detail |
|---|---|
| `calculate_book_confidence()` | Scores title, author, publisher, date, isbn (0.0–1.0) |
| `calculate_magazine_confidence()` | Scores magazine_name, editor, publisher, issue_date, issue_number, price |

### 5.14 `models.py` — Pydantic Data Models

| Model | Fields |
|---|---|
| `BookMetadata` | title, author, publisher, isbn, published_date, confidence |
| `MagazineMetadata` | magazine_name, editor, publisher, issue_date, issue_number, price, language, confidence |
| `ImageMetadata` | width, height, format, color_space, bit_depth, EXIF fields, GPS, DPI |
| `ImageVLMMetadata` | description, text_content, language, scene_classification, entities |
| `ExtractionResult` | book_metadata, magazine_metadata, image_metadata, image_vlm_metadata |
| `BenchmarkResult` | wraps ExtractionResult + debug info |

### 5.15 `prompts.py` — LLM Prompts

| Prompt | Purpose |
|---|---|
| `BOOK_EXTRACTION_PROMPT` | Extract title, author, publisher, published_date from OCR text |
| `MAGAZINE_EXTRACTION_PROMPT` | Extract magazine_name, editor, publisher, issue_date, issue_number, price |
| `IMAGE_ANALYSIS_PROMPT` | Analyze image → description, text_content, language, scene_classification, entities |
| `LANGUAGE_LABELS` | Maps lang codes to human-readable labels (te, en, hi) |

### 5.16 `hardware.py` — Hardware Detection

| Detection Order | Detail |
|---|---|
| TPU → NVIDIA (torch) → NVIDIA (NVML) → Apple MPS → CPU | Auto-configures dtype, memory_util, tensor_parallel_size |

### 5.17 `models_registry.py` — Model Cache Management

| Function | Detail |
|---|---|
| `AVAILABLE_MODELS` | List of registered models with VRAM requirements |
| `is_model_cached()` | Checks HF cache for model |
| `get_cached_models()` | Returns list of cached model IDs |
| `get_cached_model_size()` | Returns disk size of cached model |
| `remove_model_from_cache()` | Deletes model from HF cache |

### 5.18 `tasks.py` — Celery Task Definitions

| Task | Queue | LLM |
|---|---|---|
| `extract_pdf_task` | `vlm_queue` | Yes |
| `extract_image_task` | `default_queue` or `vlm_queue` | No (VLM optional) |
| `extract_text_task` | `vlm_queue` | Yes |

### 5.19 `celery_config.py` — Celery Configuration

| Setting | Value |
|---|---|
| `broker_url` | `redis://localhost:6379/0` |
| `result_backend` | `redis://localhost:6379/0` |
| `task_routes` | PDF/text → `vlm_queue`, images → runtime routing |

---

## 6. Supported File Extensions

| Extension | Processing | LLM Required | VLM Required |
|---|---|---|---|
| `.pdf` | vParse OCR → LLM extraction | Yes | No |
| `.md` | Plain text → LLM extraction | Yes | No |
| `.json` | Parse segments/transcription → LLM extraction | Yes | No |
| `.txt` | Plain text → LLM extraction | Yes | No |
| `.jpg` / `.jpeg` | EXIF extraction (+ optional VLM) | No | Optional (`--vlm`) |
| `.png` | EXIF extraction (+ optional VLM) | No | Optional (`--vlm`) |
| `.webp` | EXIF extraction (+ optional VLM) | No | Optional (`--vlm`) |
| `.tiff` / `.tif` | EXIF extraction (+ optional VLM) | No | Optional (`--vlm`) |

---

## 7. CLI Commands

```
bookextractor extract <input> <output.json> [options]
bookextractor api [--host 0.0.0.0] [--port 8000]
bookextractor worker [--queue default_queue] [--concurrency 4]
bookextractor hardware-info
bookextractor model list
bookextractor model download
bookextractor model remove <query>
bookextractor model cache
```

### Extract Options

| Flag | Default | Purpose |
|---|---|---|
| `--lang` | `en` | OCR language: `en`, `te`, `devanagari` |
| `--vlm` | `False` | Enable VLM image analysis (requires GPU) |
| `--benchmark` | `False` | Include debug info in output |
| `--max-model-len` | `4096` | Context window size (reduce to save VRAM) |
| `--backend` / `-b` | `auto` | LLM backend: `auto` (env-driven), `remote` (force anyLLM), `local` (force vLLM) |
