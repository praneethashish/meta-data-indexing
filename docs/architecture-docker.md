# BookExtractor — Docker Mode Architecture

**Branch:** `feat/anyllm-integration`
**Date:** 2026-05-22

---

## 1. System Overview

In Docker mode, BookExtractor runs as a multi-service deployment with FastAPI, Celery workers, Redis, and optional vParse OCR services. The key difference from CLI mode is **background task processing** — extraction jobs are queued, processed asynchronously, and results are polled.

### LLM Backend in Docker

| Configuration | Text Extraction | Image VLM | GPU Required |
|---|---|---|---|
| anyLLm env vars set in `.env` | `AnyLLMClient` (remote HTTP) | ❌ Skipped (EXIF only) | No |
| No anyLLm vars, local model downloaded | `LocalVLLMClient` (vLLM on GPU) | `ModelClient` (vLLM on GPU) | Yes |

**Important:** anyLLm is text-only. Even with anyLLm configured, `describe_image()` requires a local vLLM model on GPU. If no GPU is available, images get EXIF metadata only.

**Backend Override:** Workers can be configured with a specific backend mode via `llm_backend` parameter in `ExtractionPipeline`. Options: `auto` (env-driven, default), `remote` (force anyLLM), `local` (force vLLM).

---

## 2. Architecture Diagram

### 2.1 Docker Mode — Overall System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         docker-compose.yml                                    │
│                                                                               │
│  ┌─────────────────┐                                                          │
│  │     CLIENT       │  curl / POST / browser                                  │
│  └────────┬─────────┘                                                          │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────┐         ┌──────────────┐                                 │
│  │  metaextractor  │         │    redis     │                                 │
│  │  (FastAPI)      │◄───────►│  6379        │                                 │
│  │  port 8000      │  tasks  │  broker      │                                 │
│  │  No GPU         │  results│  backend     │                                 │
│  └────────┬────────┘         └──────┬───────┘                                 │
│           │                         │                                          │
│           │                    ┌────┴────┐                                    │
│           │                    ▼         ▼                                    │
│           │           ┌────────────┐ ┌────────────┐                          │
│           │           │ worker-cpu │ │ worker-gpu │                          │
│           │           │ default_q  │ │ vlm_queue  │                          │
│           │           │ conc: 8    │ │ conc: 2    │                          │
│           │           │ No GPU     │ │ NVIDIA GPU │                          │
│           │           └────────────┘ └─────┬──────┘                          │
│           │                                │                                  │
│           │    ┌───────────────────────────┘                                  │
│           │    │                                                              │
│           │    ▼                                                              │
│  ┌────────┴────────────────────────────────────────────────────────┐        │
│  │                    model-downloader                              │        │
│  │                    Downloads HF model to shared cache            │        │
│  │                    (runs once, then exits)                       │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │              VParse Services (optional, via profiles)            │        │
│  │  vparse-pipeline [profile: pipeline]  — CPU-only OCR            │        │
│  │  vparse-vlm      [profile: vlm]       — GPU-accelerated OCR     │        │
│  │  vparse-hybrid   [profile: hybrid]    — Full pipeline + VLM     │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  Volumes:                                                                     │
│    ~/.cache/huggingface:/models/huggingface  ← Shared HF cache                │
│    models:/models                                ← Named volume               │
│    ./uploads:/app/uploads                        ← Persistent uploads         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Async Extraction Flow (PDF/Text)

```
                    Client
                      │
                      │ POST /extract/async (multipart/form-data)
                      ▼
            ┌─────────────────────┐
            │  metaextractor      │
            │  (FastAPI)          │
            │                     │
            │  1. Validate ext    │
            │  2. Save to uploads/│
            │     {uuid}_{name}   │
            │  3. task.delay()    │
            │  4. Return job_id   │
            └─────────┬───────────┘
                      │
                      │ extract_pdf_task / extract_text_task
                      ▼
            ┌─────────────────────┐
            │  redis              │
            │  (broker)           │
            │  queues to          │
            │  vlm_queue          │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  worker-gpu         │
            │                     │
            │  get_pipeline(True) │
            │  process_*_sync()   │
            └─────────┬───────────┘
                      │
              ┌───────┼───────────────┐
              ▼       ▼               ▼
        ┌────────┐ ┌──────┐   ┌──────────────┐
        │ vParse │ │ LLM  │   │ Open Library │
        │ OCR    │ │      │   │ ISBN lookup  │
        │ HTTP   │ │      │   │ HTTP         │
        └────────┘ └──┬───┘   └──────────────┘
                      │
              ┌───────┴───────┐
              │               │
              ▼               ▼
        AnyLLMClient    LocalVLLMClient
        (remote HTTP)   (vLLM on GPU)
              │               │
              └───────┬───────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  redis              │
            │  (result backend)   │
            │  stores result      │
            └─────────────────────┘
                      ▲
                      │
            Client polls GET /jobs/{id}
```

### 2.3 Async Image Flow (No VLM)

```
                    Client
                      │
                      │ POST /extract/async (multipart/form-data)
                      ▼
            ┌─────────────────────┐
            │  metaextractor      │
            │  (FastAPI)          │
            │                     │
            │  1. Save to uploads/│
            │  2. task.apply_async│
            │     (default_queue) │
            │  3. Return job_id   │
            └─────────┬───────────┘
                      │
                      │ extract_image_task
                      ▼
            ┌─────────────────────┐
            │  redis              │
            │  queues to          │
            │  default_queue      │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  worker-cpu         │
            │                     │
            │  get_pipeline(False)│
            │  process_image_sync │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  extract_image_     │
            │  metadata()         │
            │                     │
            │  PIL + piexif       │
            │  → EXIF, GPS, DPI   │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  redis              │
            │  stores result      │
            └─────────────────────┘
```

### 2.4 Async Image Flow (With VLM)

```
                    Client
                      │
                      │ POST /extract/async (use_vlm=true)
                      ▼
            ┌─────────────────────┐
            │  metaextractor      │
            │  (FastAPI)          │
            │                     │
            │  1. Save to uploads/│
            │  2. task.apply_async│
            │     (vlm_queue)     │
            │  3. Return job_id   │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  redis              │
            │  queues to          │
            │  vlm_queue          │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  worker-gpu         │
            │                     │
            │  get_pipeline(True) │
            │  vision_client =    │
            │  ModelClient.get()  │
            │  └─► loads vLLM GPU │
            └─────────┬───────────┘
                      │
              ┌───────┴───────┐
              ▼               ▼
        ┌──────────┐   ┌──────────┐
        │ EXIF     │   │ VLM      │
        │ PIL+piex │   │ describe │
        │          │   │ _image() │
        └──────────┘   └────┬─────┘
                            │
                            ▼
                      vLLM engine
                      (GPU inference)
                            │
                      ┌─────┴─────┐
                      ▼           ▼
                description    entities
                text_content   scene_class
                language
```

### 2.5 Sync Image Flow (FastAPI, No Queue)

```
                    Client
                      │
                      │ POST /extract (multipart/form-data)
                      ▼
            ┌─────────────────────┐
            │  metaextractor      │
            │  (FastAPI)          │
            │                     │
            │  1. Validate: images│
            │     only            │
            │  2. Save to temp    │
            │  3. get_vision_     │
            │     pipeline()      │
            │  4. process_image() │
            │  5. Return result   │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  ExtractionPipeline │
            │  load_vlm=False     │
            │                     │
            │  vision_client=None │
            │  llm_client=None    │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  extract_image_     │
            │  metadata()         │
            │  → EXIF only        │
            │  (no VLM, no LLM)   │
            └─────────────────────┘
```

### 2.6 LLM Backend Selection in Workers

```
               worker-gpu / worker-cpu
                       │
                       ▼
               get_pipeline(load_vlm)
                       │
                       ▼
               ExtractionPipeline.__init__(llm_backend=...)
                       │
               ┌───────┼───────┐
               │               │
               ▼               ▼
         llm_client        vision_client
         (lazy)            (if load_vlm=True)
               │               │
               ▼               ▼
         ensure_llm_       ModelClient
         client()          .get_instance()
               │               │
               ▼               ▼
         create_llm_       vLLM engine
         client(backend)   (GPU)
               │
       ┌───────┼───────┐
       │       │       │
    "local" "remote" "auto"
       │       │       │
       ▼       │       ▼
 LocalVLLM    │   has_partial?
 Client       │   ├── Yes → Error
       │       │   └── No  → has_remote?
       │       ▼               │
       │   AnyLLMClient        ├── Yes → AnyLLMClient
       │   (remote HTTP)       └── No  → LocalVLLMClient
       │       │
       └───────┘
```

---

## 3. Services

```
┌─────────────────────────────────────────────────────────────────────┐
│                        docker-compose.yml                            │
│                                                                       │
│  ┌─────────────┐    ┌──────────┐    ┌──────────────┐                │
│  │ metaextractor│    │  redis   │    │model-downloader│              │
│  │ (FastAPI)   │    │  6379    │    │ (HF download) │                │
│  │ port 8000   │    │          │    │               │                │
│  └──────┬──────┘    └────┬─────┘    └───────┬───────┘                │
│         │                │                   │                         │
│  ┌──────┴──────┐         │                   │                         │
│  │ worker-gpu  │         │                   │                         │
│  │ vlm_queue   │◄────────┘                   │                         │
│  │ concurrency 2│                            │                         │
│  │ NVIDIA GPU  │                             │                         │
│  └─────────────┘                             │                         │
│  ┌─────────────┐                             │                         │
│  │ worker-cpu  │                             │                         │
│  │ default_queue│                            │                         │
│  │ concurrency 8│                            │                         │
│  │ CPU only    │                             │                         │
│  └─────────────┘                             │                         │
│                                               │                         │
│  ┌─────────────────────────────────────────────┘                         │
│  │ (VParse services - optional profiles)                                 │
│  │  vparse-pipeline  [profile: pipeline]                                 │
│  │  vparse-vlm       [profile: vlm]                                      │
│  │  vparse-hybrid    [profile: hybrid]                                   │
│  └──────────────────────────────────────────────────────────────────────┘
```

### 3.1 `metaextractor` — FastAPI Server

| Property | Value |
|---|---|
| Image | `metaextractor:latest` |
| Port | `8000:8000` |
| Command | `metaextractor --api` (default) |
| GPU | No |
| Purpose | Receives file uploads, queues Celery tasks, serves job status |

**Endpoints:**
- `GET /health` — Health check + model readiness
- `POST /extract` — Sync image extraction (EXIF only, no VLM)
- `POST /extract/async` — Async extraction (PDF, text, images)
- `GET /jobs/{job_id}` — Poll job status/result

### 3.2 `redis` — Message Broker + Result Backend

| Property | Value |
|---|---|
| Image | `redis:7-alpine` |
| Port | `6379:6379` |
| Purpose | Task queue (broker) + result storage (backend) |

### 3.3 `worker-gpu` — GPU Celery Worker

| Property | Value |
|---|---|
| Image | `metaextractor:latest` |
| Command | `metaextractor worker --queue vlm_queue --concurrency 2` |
| GPU | Yes (NVIDIA, 1 device) |
| Purpose | Processes PDF and text extraction tasks (LLM inference) |

### 3.4 `worker-cpu` — CPU Celery Worker

| Property | Value |
|---|---|
| Image | `metaextractor:latest` |
| Command | `metaextractor worker --queue default_queue --concurrency 8` |
| GPU | No |
| Purpose | Processes image extraction tasks (EXIF only, no LLM) |

### 3.5 `model-downloader` — HuggingFace Model Downloader

| Property | Value |
|---|---|
| Image | Custom (`Dockerfile.model-downloader`) |
| Purpose | Downloads vLLM model to shared HF cache before workers start |
| Run Once | `condition: service_completed_successfully` |

### 3.6 VParse Services (Optional, via Profiles)

| Service | Profile | GPU | Purpose |
|---|---|---|---|
| `vparse-pipeline` | `pipeline` | No | CPU-only OCR (slower, no GPU needed) |
| `vparse-vlm` | `vlm` | Yes | GPU-accelerated OCR with VLM |
| `vparse-hybrid` | `hybrid` | Yes | Full pipeline + VLM models |

---

## 4. Volume Mount Strategy

```
~/.cache/huggingface:/models/huggingface   ← Shared HF cache (bind mount)
models:/models                              ← Named volume for models
./uploads:/app/uploads                      ← Persistent uploads directory
```

All services share the same HuggingFace cache so models downloaded by `model-downloader` are visible to `metaextractor`, `worker-gpu`, and all vParse services.

---

## 5. Environment Variables

### Shared (all services)

| Variable | Value | Purpose |
|---|---|---|
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Redis broker |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Redis result backend |
| `METAEXTRACTOR_UPLOAD_DIR` | `/app/uploads` | Upload directory inside container |
| `HF_HOME` | `/models/huggingface` | HuggingFace cache path |

### anyLLm (optional, for remote LLM)

| Variable | Example | Purpose |
|---|---|---|
| `METAEXTRACTOR_LLM_MODEL` | `qwen2.5:7b` | Remote model name |
| `METAEXTRACTOR_LLM_BASE_URL` | `http://ollama-host:11434/v1` | Remote endpoint |
| `METAEXTRACTOR_LLM_API_KEY` | `ollama` | API key |
| `METAEXTRACTOR_LLM_PROVIDER` | `openai` | Provider (default: openai) |

### Local vLLM (fallback when anyLLm not configured)

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model ID to download |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | GPU count for tensor parallelism |

### vParse

| Variable | Value | Purpose |
|---|---|---|
| `VPARSE_API_URL` | `http://vparse:8000/file_parse` | vParse OCR endpoint |

---

## 6. Data Flow by File Extension (Docker Mode)

### 6.1 `.pdf` Files (Async)

```
Client → POST /extract/async (multipart/form-data)
    │
    ▼ metaextractor (FastAPI)
    │   1. Validate file extension
    │   2. Save to /app/uploads/{uuid}_{filename}
    │   3. extract_pdf_task.delay(save_path, lang="en")
    │      └── Redis broker queues to vlm_queue
    │   4. Return: {"job_id": "...", "status": "submitted"}
    │
    ▼ Redis → vlm_queue
    │
    ▼ worker-gpu picks up task
    │   1. get_pipeline(load_vlm=True)  [lazy singleton per worker]
    │   2. process_pdf_sync(pdf_path, lang="en")
    │      ├── parse_pdf_via_vparse(pdf_path, lang)
    │      │   └── HTTP POST to http://vparse:8000/file_parse
    │      │   └── Returns content_list[] or md_content
    │      ├── extract_from_text(full_text)
    │      │   ├── detect_content_type(text) → "book" or "magazine"
    │      │   ├── ensure_llm_client()
    │      │   │   └── create_llm_client()
    │      │   │       ├── anyLLm configured? → AnyLLMClient (HTTP to remote)
    │      │   │       └── No? → LocalVLLMClient (vLLM on GPU)
    │      │   ├── extract_semantic_fields(text) → LLM returns JSON
    │      │   ├── extract_isbn_candidates(text) → regex
    │      │   ├── lookup_isbn(isbn) → HTTP to openlibrary.org
    │      │   └── calculate_book_confidence()
    │      └── Return ExtractionResult
    │   3. Celery stores result in Redis
    │   4. Cleanup: delete uploaded file
    │
    ▼ Client polls GET /jobs/{job_id}
    │   Returns: {"job_id": "...", "status": "SUCCESS", "ready": true, "result": {...}}
```

### 6.2 `.json` Files (Pre-OCR'd, Async)

```
Client → POST /extract/async
    │
    ▼ metaextractor (FastAPI)
    │   1. Save to /app/uploads/{uuid}_{filename}
    │   2. extract_text_task.delay(save_path, lang="en")
    │      └── Redis broker queues to vlm_queue
    │   3. Return: {"job_id": "...", "status": "submitted"}
    │
    ▼ worker-gpu picks up task
    │   1. get_pipeline(load_vlm=True)
    │   2. process_text_file_sync(file_path, lang="en")
    │      ├── Read JSON → json.loads()
    │      ├── Check "transcription" → use it
    │      ├── Check "segments" or "content_list" → sort by reading_order → join .text
    │      ├── Fallback: json.dumps(entire JSON)
    │      ├── extract_from_text(clean_text)
    │      │   └── Same LLM flow as PDF (see 5.1)
    │      └── Return ExtractionResult
    │   3. Celery stores result in Redis
    │   4. Cleanup: delete uploaded file
    │
    ▼ Client polls GET /jobs/{job_id}
```

### 6.3 `.md` / `.txt` Files (Async)

```
Client → POST /extract/async
    │
    ▼ metaextractor (FastAPI)
    │   1. Save to /app/uploads/{uuid}_{filename}
    │   2. extract_text_task.delay(save_path, lang="en")
    │      └── Redis broker queues to vlm_queue
    │   3. Return: {"job_id": "...", "status": "submitted"}
    │
    ▼ worker-gpu picks up task
    │   1. get_pipeline(load_vlm=True)
    │   2. process_text_file_sync(file_path)
    │      ├── Read file as plain text
    │      ├── extract_from_text(content)
    │      │   └── Same LLM flow as PDF (see 5.1)
    │      └── Return ExtractionResult
    │   3. Celery stores result in Redis
    │   4. Cleanup: delete uploaded file
```

### 6.4 `.jpg` / `.png` / `.webp` / `.tiff` Files (Async)

```
Client → POST /extract/async (use_vlm=false by default)
    │
    ▼ metaextractor (FastAPI)
    │   1. Save to /app/uploads/{uuid}_{filename}
    │   2. extract_image_task.apply_async(queue="default_queue")
    │      └── Redis broker queues to default_queue
    │   3. Return: {"job_id": "...", "status": "submitted"}
    │
    ▼ worker-cpu picks up task
    │   1. get_pipeline(load_vlm=False)  [no LLM, no GPU]
    │   2. process_image_sync(image_path)
    │      ├── extract_image_metadata(image_path)
    │      │   └── PIL + piexif → EXIF, dimensions, GPS, DPI
    │      └── vision_client is None → skip VLM analysis
    │   3. Celery stores result in Redis
    │   4. Cleanup: delete uploaded file
    │
    ▼ Client polls GET /jobs/{job_id}
```

### 6.5 Images with VLM (Async, requires GPU)

```
Client → POST /extract/async (use_vlm=true)
    │
    ▼ metaextractor (FastAPI)
    │   1. Save to /app/uploads/{uuid}_{filename}
    │   2. extract_image_task.apply_async(queue="vlm_queue")
    │      └── Redis broker queues to vlm_queue
    │   3. Return: {"job_id": "...", "status": "submitted"}
    │
    ▼ worker-gpu picks up task
    │   1. get_pipeline(load_vlm=True)
    │      └── vision_client = ModelClient.get_instance() → loads vLLM on GPU
    │   2. process_image_sync(image_path, use_vlm=True)
    │      ├── extract_image_metadata(image_path) → EXIF
    │      ├── vision_client.describe_image(image_path) → VLM analysis
    │      │   └── Returns: description, text_content, language, scene_classification, entities
    │      └── Return ExtractionResult(image_metadata, image_vlm_metadata)
    │   3. Celery stores result in Redis
    │   4. Cleanup: delete uploaded file
```

### 6.6 Sync Image Extraction (FastAPI, no queue)

```
Client → POST /extract (multipart/form-data)
    │
    ▼ metaextractor (FastAPI)
    │   1. Validate: only images supported for sync
    │   2. Save to temp directory (auto-cleanup)
    │   3. get_vision_pipeline() → ExtractionPipeline(load_vlm=False)
    │   4. process_image(temp_path)
    │      └── EXIF only (no VLM, no LLM)
    │   5. Return result immediately
```

---

## 7. Queue Routing

| Task | Default Queue | GPU Required | LLM Used |
|---|---|---|---|
| `extract_pdf` | `vlm_queue` | Yes (or anyLLm remote) | Yes |
| `extract_text` | `vlm_queue` | Yes (or anyLLm remote) | Yes |
| `extract_image` (no VLM) | `default_queue` | No | No |
| `extract_image` (with VLM) | `vlm_queue` | Yes | Yes (vision only) |

---

## 8. Docker Compose Commands

### Start all services (no vParse)

```bash
docker compose up -d
```

### Start with vParse (GPU required)

```bash
docker compose --profile vlm up -d
```

### Start with vParse (CPU only, slower)

```bash
docker compose --profile pipeline up -d
```

### Start with full vParse (pipeline + VLM)

```bash
docker compose --profile hybrid up -d
```

### View logs

```bash
docker compose logs -f metaextractor
docker compose logs -f worker-gpu
docker compose logs -f worker-cpu
```

### Stop all services

```bash
docker compose down
```

### Stop and remove volumes

```bash
docker compose down -v
```

---

## 9. Supported File Extensions (Docker Mode)

| Extension | Processing | Queue | Worker | LLM Required |
|---|---|---|---|---|
| `.pdf` | vParse OCR → LLM extraction | `vlm_queue` | `worker-gpu` | Yes |
| `.md` | Plain text → LLM extraction | `vlm_queue` | `worker-gpu` | Yes |
| `.json` | Parse segments/transcription → LLM extraction | `vlm_queue` | `worker-gpu` | Yes |
| `.txt` | Plain text → LLM extraction | `vlm_queue` | `worker-gpu` | Yes |
| `.jpg` / `.jpeg` | EXIF extraction (+ optional VLM) | `default_queue` or `vlm_queue` | `worker-cpu` or `worker-gpu` | No (VLM optional) |
| `.png` | EXIF extraction (+ optional VLM) | `default_queue` or `vlm_queue` | `worker-cpu` or `worker-gpu` | No (VLM optional) |
| `.webp` | EXIF extraction (+ optional VLM) | `default_queue` or `vlm_queue` | `worker-cpu` or `worker-gpu` | No (VLM optional) |
| `.tiff` / `.tif` | EXIF extraction (+ optional VLM) | `default_queue` or `vlm_queue` | `worker-cpu` or `worker-gpu` | No (VLM optional) |

---

## 10. anyLLm in Docker — Complete Flow

When `.env` contains:
```env
METAEXTRACTOR_LLM_MODEL=qwen2.5:7b
METAEXTRACTOR_LLM_BASE_URL=http://ollama-host:11434/v1
METAEXTRACTOR_LLM_API_KEY=ollama
```

### What Changes

| Component | Without anyLLm | With anyLLm |
|---|---|---|
| `worker-gpu` text extraction | vLLM on GPU (needs model download) | HTTP to remote Ollama |
| `worker-gpu` GPU usage | High (model loading + inference) | Low (only vision tasks) |
| `model-downloader` | Required (downloads vLLM model) | Optional (can skip) |
| `worker-cpu` | No change | No change |
| Image VLM analysis | Works (local vLLM) | ❌ Skipped (EXIF only) |

### Docker Setup with anyLLm Only (No GPU)

```yaml
# docker-compose.yml — GPU-free setup
services:
  metaextractor:
    build: .
    ports: ["8000:8000"]
    env_file: .env  # Contains METAEXTRACTOR_LLM_* vars
    # No GPU, no model-downloader needed

  redis:
    image: redis:7-alpine

  worker-cpu:
    build: .
    command: metaextractor worker --queue default_queue --concurrency 8
    env_file: .env
    # Processes images (EXIF only) — no LLM needed
```

**Note:** Without GPU, PDF/text extraction still works via anyLLm (remote HTTP), but image VLM analysis is unavailable.
