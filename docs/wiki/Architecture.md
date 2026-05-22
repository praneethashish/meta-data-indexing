# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │   CLI (Typer) │  │  FastAPI     │  │  Celery Worker (CLI)      │  │
│  │  extract     │  │  /extract    │  │  worker -q ...            │  │
│  │  model list  │  │  /extract/async                              │  │
│  │  model dl    │  │  /jobs/{id}  │  │                           │  │
│  └──────┬───────┘  └──────┬───────┘  └─────────────┬──────────────┘  │
│         │                 │                         │                  │
└─────────┼─────────────────┼─────────────────────────┼──────────────────┘
          │                 │                         │
          ▼                 ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                              │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                      ExtractionPipeline                         │  │
│  │                                                                 │  │
│  │  process_pdf() ──► vParse OCR ──► extract_from_text()         │  │
│  │  process_image() ──► extract_image_metadata()                 │  │
│  │  process_text_file() ──► extract_from_text()                  │  │
│  │                                                                 │  │
│  │  extract_from_text() ─┬─► _extract_book_metadata()            │  │
│  │                       └─► _extract_magazine_metadata()        │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           │                                           │
│              ┌────────────┼────────────┐                              │
│              ▼            ▼            ▼                              │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐                    │
│  │  llm_client  │ │ validation │ │ external_api │                    │
│  │  (lazy init) │ │  (ISBN)    │ │ (OpenLibrary)│                    │
│  │  AnyLLM or   │ │            │ │              │                    │
│  │  LocalVLLM   │ │            │ │              │                    │
│  └──────────────┘ └────────────┘ └──────────────┘                    │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐ ┌───────────────┐ │
│  │ vision_client│ │ models.py  │ │ models_reg.  │ │ vparse_client │ │
│  │ (ModelClient)│ │ Pydantic   │ │ HF cache     │ │ HTTP to vParse│ │
│  │ (GPU, vLLM)  │ │ Data models│ │ scanner      │ │ OCR API       │ │
│  └──────────────┘ └────────────┘ └──────────────┘ └───────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │                                                 │
          ▼                                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ASYNC / QUEUE LAYER                            │
│                                                                       │
│  ┌─────────────────┐          ┌─────────────────┐                     │
│  │  Redis Broker   │◄────────►│  Celery Workers  │                     │
│  │  redis:6379/0   │          │                  │                     │
│  │  (task queue)   │          │  worker-gpu      │                     │
│  │  (result backend)│         │  - vlm_queue     │                     │
│  └─────────────────┘          │  - concurrency 2 │                     │
│                               │  - NVIDIA GPU    │                     │
│                               │                  │                     │
│                               │  worker-cpu      │                     │
│                               │  - default_queue │                     │
│                               │  - concurrency 8 │                     │
│                               │  - CPU only      │                     │
│                               └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SERVICES                              │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │  vParse API  │  │  HuggingFace │  │  Remote LLM (anyLLM)       │  │
│  │  (mineru)    │  │  Model Cache │  │  (Ollama, HF, OpenAI)      │  │
│  │  OCR engine  │  │  ~/.cache/   │  │  HTTP POST /v1/chat        │  │
│  │  port 8000   │  │  huggingface │  │                            │  │
│  └──────────────┘  └──────────────┘  └────────────────────────────┘  │
│  ┌──────────────────────────────────┐                                │
│  │  Open Library API                │                                │
│  │  (ISBN validation)               │                                │
│  │  https://openlibrary.org         │                                │
│  └──────────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Deep Modules

| Module                  | File                        | Responsibility                   | Dependencies                                        |
| ----------------------- | --------------------------- | -------------------------------- | --------------------------------------------------- |
| **Format Router**       | `main.py`                   | Route files by extension, LLM backend selection | pipeline, tasks, llm_clients |
| **Extraction Pipeline** | `pipeline.py`               | Orchestrate extraction           | vparse_client, llm_clients, model_client, external_api, validation |
| **LLM Clients**         | `llm_clients.py`            | LLM abstraction (remote/local)   | anyllm, model_client, model_manager                 |
| **Model Client**        | `model_client.py`           | vLLM singleton wrapper           | vllm, hardware                                      |
| **Image Utils**         | `image_utils.py`            | EXIF extraction                  | PIL, piexif                                         |
| **Validation**          | `validation.py`             | ISBN validation                  | regex                                               |
| **Hardware Detection**  | `hardware.py`               | GPU/CPU/TPU detection            | pynvml, torch, psutil                               |
| **Models Registry**     | `models_registry.py`        | Model cache scanner              | huggingface_hub                                     |
| **Celery Tasks**        | `tasks.py`                  | Async task definitions           | pipeline, celery                                    |

### Shallow Modules (Adapters)

| Module            | File               | Responsibility      | Dependencies |
| ----------------- | ------------------ | ------------------- | ------------ |
| **VParse Client** | `vparse_client.py` | VParse API adapter  | httpx        |
| **External API**  | `external_api.py`  | OpenLibrary adapter | httpx        |

---

## Data Flow

### PDF Processing Flow

```
PDF Upload
    │
    ▼
┌───────────────────────────────┐
│ vParse OCR API                │
│ - Layout detection            │
│ - Text extraction             │
│ - Returns content_list        │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Text Processing               │
│ - Clean text                  │
│ - Join content_list           │
│ - Remove image refs           │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Content Type Detection        │
│ - Magazine vs Book            │
└───────────────────────────────┘
    │
    ├── Book ──────────────────────┐
    │                              │
    ▼                              ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Book Metadata Extraction      │ │ Magazine Metadata Extraction  │
│ - LLM semantic fields         │ │ - LLM magazine prompt         │
│ - ISBN regex extraction       │ │ - Dynamic language labels     │
│ - OpenLibrary lookup          │ │ - Confidence scoring          │
│ - Confidence scoring          │ │                               │
└──────────────┬────────────────┘ └──────────────┬────────────────┘
               │                                  │
               └──────────────┬───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  JSON Result    │
                    └─────────────────┘
```

### Image Processing Flow

```
Image Upload
    │
    ▼
┌───────────────────────────────┐
│ EXIF Extraction (PIL/piexif)  │
│ - Width, height, format       │
│ - Camera make/model           │
│ - GPS coordinates (DMS→decimal)│
│ - Date, lens info, DPI        │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│  JSON Result (ImageMetadata)  │
└───────────────────────────────┘
```

### Async Processing Flow

```
POST /extract/async
    │
    ▼
┌───────────────────────────────┐
│ Save file to uploads/         │
│ Generate UUID file_id         │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Route to Celery task          │
│ - PDF → extract_pdf_task      │
│ - Image → extract_image_task  │
│ - Text → extract_text_task    │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Redis broker queues task      │
│ - vlm_queue (GPU) for PDF/text│
│ - default_queue (CPU) for img │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Worker picks up task          │
│ - Lazy-load pipeline singleton│
│ - Execute extraction          │
│ - Cleanup uploaded file       │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Redis stores result           │
│ Client polls GET /jobs/{id}   │
└───────────────────────────────┘
```

---

## Technology Stack

| Layer                | Technology                      | Purpose                        |
| -------------------- | ------------------------------- | ------------------------------ |
| **API**              | FastAPI                         | REST endpoints, async          |
| **CLI**              | Typer                           | Command-line interface         |
| **OCR**              | VParse (mineru-dots)            | PDF/document OCR               |
| **LLM (Remote)**     | anyLLM                          | HTTP-based inference (Ollama, HF, OpenAI) |
| **LLM (Local)**      | Qwen2.5-VL, Qwen3-VL, Gemma 4   | Text semantic extraction       |
| **Inference Engine** | vLLM + PyTorch                  | GPU-accelerated LLM inference  |
| **ISBN Lookup**      | OpenLibrary API                 | Book metadata                  |
| **Tasks**            | Celery + Redis                  | Async processing               |
| **Container**        | Docker                          | Deployment                     |

---

## External Dependencies

### APIs

| Service     | Purpose         | Rate Limit  |
| ----------- | --------------- | ----------- |
| VParse API  | OCR             | N/A (local) |
| OpenLibrary | ISBN enrichment | 100 req/s   |

---

## Configuration

### Environment Variables

| Variable                      | Description                        | Default                            |
| ----------------------------- | ---------------------------------- | ---------------------------------- |
| `VPARSE_API_URL`              | VParse API endpoint                | `http://localhost:8000/file_parse` |
| `BOOKEXTRACTOR_LLM_MODEL`     | Remote model identifier            | _(unset — uses local vLLM)_        |
| `BOOKEXTRACTOR_LLM_BASE_URL`  | Remote API base URL                | _(unset)_                          |
| `BOOKEXTRACTOR_LLM_API_KEY`   | Remote API key                     | _(unset)_                          |
| `BOOKEXTRACTOR_LLM_PROVIDER`  | anyLLM provider name               | `openai`                           |
| `VLLM_MODEL_ID`               | HuggingFace model ID               | `Qwen/Qwen2.5-VL-7B-Instruct`      |
| `CELERY_BROKER_URL`           | Redis broker URL                   | `redis://localhost:6379/0`         |
| `CELERY_RESULT_BACKEND`       | Redis result backend               | `redis://localhost:6379/0`         |
| `BOOKEXTRACTOR_UPLOAD_DIR`    | Upload directory                   | `uploads`                          |
| `VLLM_DEVICE`                 | Target device: cuda, tpu, mps, cpu | Auto-detected                      |
| `VLLM_DTYPE`                  | Model precision                    | Auto-optimized                     |
| `VLLM_GPU_MEMORY_UTILIZATION` | GPU memory fraction (0.0-1.0)      | Auto-optimized                     |
| `VLLM_TENSOR_PARALLEL_SIZE`   | Number of GPUs (or `auto`)         | Auto-optimized                     |

### CLI Backend Flag

| Flag | Values | Purpose |
| ---- | ------ | ------- |
| `--backend` / `-b` | `auto` (default) | Env-driven: remote vars set → anyLLM, else → vLLM |
| | `remote` | Force anyLLM (requires `BOOKEXTRACTOR_LLM_*` vars) |
| | `local` | Force local vLLM (ignores remote env vars) |

---

## Security

### Input Validation

- File type validation by extension
- Malformed file handling

### API Security

- API key authentication (future)
- Rate limiting (future)

---

## Performance Considerations

### CPU-bound Operations

- vParse OCR (external service)
- EXIF extraction (PIL + piexif)
- anyLLM remote inference (HTTP request, no local GPU)

### GPU-bound Operations

- vLLM local LLM inference (text semantic extraction, when `--backend local` or no remote config)
- VLM image analysis (`describe_image()`, always requires local vLLM)

### Memory Considerations

- Pipeline singleton per worker process
- Worker restart after 100 tasks (memory leak prevention)
- GPU memory utilization auto-optimized by hardware detection

---

## Scalability

### Horizontal Scaling

- API servers: Stateless, scale behind load balancer
- CPU workers: Celery prefork pool (concurrency 8)
- GPU workers: Separate queue for LLM tasks (concurrency 2)

### Queue Routing

| File Type | Queue | Worker | LLM | GPU Required |
|-----------|-------|--------|-----|-------------|
| PDF | `vlm_queue` | worker-gpu | Yes (anyLLM or vLLM) | Only if vLLM |
| Text/JSON/MD | `vlm_queue` | worker-gpu | Yes (anyLLM or vLLM) | Only if vLLM |
| Images | `default_queue` | worker-cpu | No | No |
| Images (with VLM) | `vlm_queue` | worker-gpu | Vision only | Yes |

---

_Last updated: Phase 4 (anyLLM integration, decoupled LLM/vision clients, --backend flag) complete_
