# Architecture

## System Overview

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      API Layer (FastAPI)                     │
                    │                                                                  │
                    │   Endpoints:                                                     │
                    │   - POST /extract          - Submit single extraction           │
                    │   - POST /extract/async    - Submit async job                  │
                    │   - POST /extract/batch    - Submit batch job                  │
                    │   - GET  /jobs/{id}        - Get job status                    │
                    │   - GET  /batch/{id}       - Get batch status                  │
                    │   - GET  /health           - Health check                      │
                    └─────────────────────────────────────────────────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────────────────┐
                    │                        ▼                                      │
                    │              ┌─────────────────┐                              │
                    │              │  Format Router  │ (main.py)                   │
                    │              │  /extract       │                              │
                    │              └────────┬────────┘                              │
                    │        ┌──────────────┼──────────────┐                        │
                    │        ▼              ▼              ▼                        │
                    │    ┌────────┐    ┌────────┐    ┌────────┐                      │
                    │    │  PDF   │    │ Image  │    │ Media  │                      │
                    │    │Pipeline│    │Pipeline│    │Pipeline│                      │
                    │    └───┬────┘    └───┬────┘    └───┬────┘                      │
                    │        │             │             │                            │
                    │        ▼             │             ▼                            │
                    │   ┌─────────┐       │      ┌───────────┐                       │
                    │   │Pre-OCR  │       │      │  ffprobe  │                       │
                    │   │ Regex   │       │      │(audio/vid)│                       │
                    │   └───┬─────┘       │      └───────────┘                       │
                    │       │             │                                            │
                    │       ▼             ▼                                            │
                    │   ┌─────────────────────────────┐                               │
                    │   │      VParse API Client      │                               │
                    │   │      (vparse_client.py)     │                               │
                    │   └──────────────┬──────────────┘                               │
                    │                  │                                              │
                     │    ┌─────────────┴─────────────┐                                │
                     │    │                          │                                │
                     │    ▼                          ▼                                │
                     │┌─────────┐              ┌──────────┐                           │
                     ││ pipeline│              │   vLLM   │                            │
                     ││ backend│              │  Gemma-4  │                            │
                     │└────┬────┘              └────┬─────┘                           │
                     │     │                        │                                  │
                     │     ▼                        ▼                                  │
                     │┌─────────────────────────────────────┐                          │
                     ││       Gemma-4 (Text + Vision)        │                          │
                     ││         via vLLM Engine              │                          │
                     │└─────────────────────────────────────┘                          │
                     │                  │                                              │
                     │                  │                                              │
                      │        ┌─────────┴──────────┐                                  │
                      │        │                    │                                   │
                      │        ▼                    ▼                                   │
                      │   ┌─────────────┐    ┌──────────────┐                          │
                      │   │  Gemma-4    │    │  OpenLibrary │                          │
                      │   │   vLLM     │    │     API      │                          │
                      │   └─────────────┘    └──────────────┘                          │
                    │         │                                                       │
                    │         └───────────────┬──────────────────────────────────────┘
                    │                         │
                    │                   ┌──────▼────────┐
                    │                   │  JSON Result  │
                    │                   └───────────────┘
                    │
                    │  ┌─────────────────────────────────────────────────────────┐
                    │  │              Celery + Redis (Phase 3)                   │
                    │  │                                                          │
                    │  │   Tasks: extract_task, batch_extract_task               │
                    │  │   Queues: default, vlm_queue, batch_queue               │
                    │  │   Results: Redis backend with 24h TTL                    │
                    │  └─────────────────────────────────────────────────────────┘
                    └──────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Deep Modules

| Module | File | Responsibility | Dependencies |
|--------|------|----------------|--------------|
| **Format Router** | `main.py` | Route files by extension | pipeline |
| **Extraction Pipeline** | `pipeline.py` | Orchestrate extraction | vparse_client, vlm_client, external_api, validation |
| **Pre-OCR Extractor** | `pdf_metadata_extractor.py` | Quick metadata before OCR | PyPDF2, regex |
| **Media Utils** | `media_utils.py` | Audio/video metadata via ffprobe | subprocess |
| **VLM Client** | `vlm_client.py` | Gemma-4 vLLM integration | httpx, vllm |
| **Image Utils** | `image_utils.py` | EXIF extraction | PIL, piexif |
| **Validation** | `validation.py` | ISBN validation | regex |
| **Celery Tasks** | `tasks.py` | Async task definitions | pipeline, celery |

### Shallow Modules (Adapters)

| Module | File | Responsibility | Dependencies |
|--------|------|----------------|--------------|
| **VParse Client** | `vparse_client.py` | VParse API adapter | httpx |
| **External API** | `external_api.py` | OpenLibrary adapter | httpx |

---

## Data Flow

### PDF Processing Flow

```
PDF Upload
    │
    ▼
┌───────────────────────────────┐
│ Pre-OCR Regex Extraction      │
│ (ISBN, title, author, etc.)   │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ VParse OCR (pipeline)         │
│ - Layout detection            │
│ - Text extraction             │
│ - OCR (PaddleOCR)             │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Text Processing               │
│ - Clean text                  │
│ - Join content_list           │
└───────────────────────────────┘
    │
    ├───┬───────────────────────┘
    │   │
    ▼   ▼
┌───────────────────────────────┐
│ LLM Semantic Extraction       │
│ (Gemma-4 via vLLM)            │
│ - Title, author, publisher    │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ ISBN Validation               │
│ - Extract ISBN from text      │
│ - OpenLibrary lookup          │
│ - Merge metadata              │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ OCR Format Standardization     │
│ - transcription, confidence   │
│ - segments, named_entities    │
└───────────────────────────────┘
    │
    ▼
JSON Result
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
│ - GPS coordinates              │
│ - Date, lens info             │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ VLM Description (Gemma-4 vLLM)│
│ - Send image to Gemma-4 vLLM  │
│ - Generate description         │
│ - Extract entities             │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Combine Results               │
│ - EXIF + VLM in single model  │
└───────────────────────────────┘
    │
    ▼
JSON Result
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **API** | FastAPI | REST endpoints, async |
| **CLI** | Typer | Command-line interface |
| **OCR** | VParse (mineru-dots) | PDF/document OCR |
| **LLM/VLM** | Gemma-4 (google/gemma-4-E4B-it) | Unified text + vision via vLLM |
| **Inference Engine** | vLLM + PyTorch | GPU-accelerated LLM inference |
| **ISBN Lookup** | OpenLibrary API | Book metadata |
| **Media** | ffprobe | Audio/video metadata |
| **Tasks** | Celery + Redis | Async processing |
| **Container** | Docker | Deployment |

---

## External Dependencies

### APIs

| Service | Purpose | Rate Limit |
|---------|---------|------------|
| VParse API | OCR and VLM | N/A (local) |
| OpenLibrary | ISBN enrichment | 100 req/s |

### External Tools

| Tool | Purpose | Required |
|------|---------|----------|
| ffprobe | Audio/video metadata | Yes |
| tesseract | Lite OCR (optional) | No |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VPARSE_API_URL` | VParse API endpoint | `http://localhost:8000/file_parse` |
| `BOOKEXTRACTOR_MODELS_DIR` | Local models directory | `./models` |
| `VLLM_MODEL` | Gemma-4 model ID | `google/gemma-4-E4B-it` |
| `HF_TOKEN` | HuggingFace token for model access | (required) |
| `CELERY_BROKER_URL` | Redis broker URL | `redis://localhost:6379/0` |
| `CELERY_RESULT_BACKEND` | Redis result backend | `redis://localhost:6379/0` |

---

## Security

### Input Validation

- File type validation by extension
- File size limits
- Malformed file handling

### API Security

- API key authentication (future: Phase N)
- Rate limiting (future: Phase 4)
- Tenant isolation (future: Phase N)

---

## Performance Considerations

### CPU-bound Operations

- VParse OCR (pipeline backend)
- Gemma-4 text processing (vLLM)
- ffprobe metadata extraction
- Pre-OCR regex extraction

### GPU-bound Operations

- Gemma-4 vLLM inference (text + vision unified)

### Memory Considerations

- Streaming for large PDFs
- Batch processing for images (Phase 4)
- GPU memory management (Phase 4)

---

## Scalability

### Horizontal Scaling

- API servers: Stateless, scale behind load balancer
- CPU workers: Celery prefork pool
- GPU workers: Separate queue for VLM tasks

### Vertical Scaling

- A100 GPU: 80GB VRAM for VLM batching
- Worker concurrency: 8 CPU workers per node

---

*Last updated: Phase 2 implementation*
