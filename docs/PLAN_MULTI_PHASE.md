# Metadata Extraction Pipeline - Multi-Phase Implementation Plan

**Last Updated:** Phase 2 revision - vLLM + PyTorch inference engine

---

## Overview

**Target Scale:** 130,000+ documents with audio/video content

**Infrastructure:** A100 GPU (80GB VRAM) for deployment, CPU for development

**Purpose:** Extract structured metadata, perform OCR, and generate AI-powered descriptions from documents and multimedia files.

---

## Technology Stack

| Component            | Technology                      | Purpose                               |
| -------------------- | ------------------------------- | ------------------------------------- |
| **API Framework**    | FastAPI                         | REST API endpoints                    |
| **CLI**              | Typer                           | Command-line interface                |
| **OCR Engine**       | VParse (mineru-dots)            | PDF and document OCR                  |
| **LLM/VLM**          | Qwen2.5-VL, Qwen3-VL, Gemma 4   | Text semantic extraction              |
| **Inference Engine** | vLLM + PyTorch                  | GPU-accelerated LLM inference         |
| **ISBN Lookup**      | OpenLibrary API                 | Book metadata enrichment              |
| **Task Queue**       | Celery + Redis                  | Async background processing           |
| **Container**        | Docker + Docker Compose         | Deployment                            |

---

## Phase 1: Foundation ✅ COMPLETE

### Implemented Features

| Feature           | Status | Description                                           |
| ----------------- | ------ | ----------------------------------------------------- |
| Format Router     | ✅     | Unified `/extract` endpoint routing by file extension |
| PDF Pipeline      | ✅     | VParse OCR API integration for document processing    |
| Text Pipeline     | ✅     | Direct LLM processing for `.md`, `.json` files        |
| Image Pipeline    | ✅     | EXIF/PIL metadata extraction from images              |
| GPS Normalization | ✅     | DMS to decimal degrees conversion                     |
| ISBN Extraction   | ✅     | Regex extraction + OpenLibrary validation             |
| Docker Setup      | ✅     | Multi-container Docker orchestration                  |
| Tests             | ✅     | Unit and integration test coverage                    |

### Components

| Module             | Path                             | Responsibility                     |
| ------------------ | -------------------------------- | ---------------------------------- |
| `main.py`          | `metaextractor/main.py`          | FastAPI application, format router |
| `pipeline.py`      | `metaextractor/pipeline.py`      | Extraction pipeline orchestrator   |
| `vparse_client.py` | `metaextractor/vparse_client.py` | VParse API client                  |
| `image_utils.py`   | `metaextractor/image_utils.py`   | EXIF extraction utilities          |
| `models.py`        | `metaextractor/models.py`        | Pydantic data models               |
| `validation.py`    | `metaextractor/validation.py`    | ISBN validation                    |
| `external_api.py`  | `metaextractor/external_api.py`  | OpenLibrary API client             |

### Environment Variables (Phase 1)

| Variable                   | Description                        | Default                            |
| -------------------------- | ---------------------------------- | ---------------------------------- |
| `VPARSE_API_URL`           | VParse API endpoint                | `http://localhost:8000/file_parse` |
| `METAEXTRACTOR_MODELS_DIR` | Local models directory             | `./models`                         |

---

## Phase 2: Multimedia & VLM Enhancement ✅ COMPLETE

### Objectives

- Switch inference engine from llama-cpp to vLLM + PyTorch
- Add hardware auto-detection and vLLM optimization
- Add model management CLI commands
- Add magazine/periodical metadata extraction
- Support multiple models (Qwen, Gemma)

### Completed

| Feature                  | Module                         | Status |
| ------------------------ | ------------------------------ | ------ |
| vLLM Engine              | `vlm_client.py`, `pipeline.py` | ✅     |
| Hardware Detection       | `hardware.py`                  | ✅     |
| Model Registry           | `models_registry.py`           | ✅     |
| Model CLI Commands       | `main.py`                      | ✅     |
| Magazine Extraction      | `pipeline.py`                  | ✅     |
| Model Downloader         | `scripts/setup_models.py`      | ✅     |
| Docker GPU Support       | `docker-compose.yml`           | ✅     |
| HF Cache (HF_HOME aware) | `models_registry.py`           | ✅     |

### Not Implemented (Deferred)

| Feature                  | Reason |
| ------------------------ | ------ |
| Audio Metadata           | No current use case |
| Video Metadata           | No current use case |
| Pre-OCR Regex            | vParse provides sufficient quality |
| Image VLM Description    | Vision model not integrated for images |
| OCR Format Standard      | vParse output is sufficient |

### 2.1 Audio/Video Metadata Extraction

**New Module:** `metaextractor/media_utils.py`

```python
# Pseudocode interface
def extract_audio_metadata(file_path: str) -> AudioMetadata:
    """Uses ffprobe to extract audio metadata"""
    # Command: ffprobe -v quiet -print_format json -show_format -show_streams <file>
    # Returns: AudioMetadata with warnings for multi-stream

def extract_video_metadata(file_path: str) -> VideoMetadata:
    """Uses ffprobe to extract video metadata"""
    # Command: ffprobe -v quiet -print_format json -show_format -show_streams <file>
    # Returns: VideoMetadata
```

**Data Models:** `models.py`

```python
class AudioMetadata(BaseModel):
    duration: float | None
    bitrate: int | None
    sample_rate: int | None
    channels: int | None
    codec: str | None
    format: str | None
    container: str | None
    size: int | None
    warnings: list[str] = []  # Multi-stream warnings

class VideoMetadata(BaseModel):
    duration: float | None
    width: int | None
    height: int | None
    codec: str | None
    framerate: float | None
    bitrate: int | None
    format: str | None
    size: int | None
```

---

### 2.2 Pre-OCR Regex Extraction

**New Module:** `metaextractor/pdf_metadata_extractor.py`

```python
def extract_pdf_metadata_pre_ocr(file_path: str) -> dict[str, Any]:
    """
    Quick metadata extraction before OCR.
    Uses PyPDF2/pdfplumber for text extraction + regex.

    Returns:
        {
            "isbn": str | None,
            "title": str | None,
            "author": str | None,
            "publisher": str | None,
            "published_date": str | None,
            "is_digital": bool  # True if text is extractable without OCR
        }
    """
```

**Regex Patterns:**

| Field      | Pattern                                                 |
| ---------- | ------------------------------------------------------- |
| ISBN-10/13 | `(?:\bISBN(?:-1[03])?:?\s*)?([0-9Xx\-\s]{10,20})`       |
| Title      | `<h1[^>]*>(.*?)</h1>` or metadata fields                |
| Author     | `by\s+(.+?)(?:\n\|,\|$)` or metadata fields             |
| Publisher  | `published\s+by\s+(.+?)(?:\n\|,\|$)` or metadata fields |

**Merge Priority:**

1. **Pre-OCR regex** (highest priority - authoritative if found)
2. **OpenLibrary** (if valid ISBN found)
3. **LLM extraction** (fallback)

---

### 2.3 Image Level 2 VLM Enhancement (Gemma-4 via vLLM)

**Module:** `metaextractor/vlm_client.py` (or integrated into `pipeline.py`)

**VLM Prompt for Gemma-4:**

```
Analyze this image extracted from a document and provide a structured response with the following:

1. DESCRIPTION: A detailed, precise description of what this image shows. If it's a document page, describe the layout, headings, and key content visible. If it's a photograph or figure, describe the scene, objects, and context.

2. TEXT_CONTENT: Extract ALL text visible in the image, preserving the reading order. Include any titles, captions, labels, or written content. If no text is present, state "No text detected".

3. LANGUAGE: Identify the primary language of any text detected (e.g., "English", "Telugu", "Hindi", "Mixed"). If no text, state "Not applicable".

4. SCENE_CLASSIFICATION: Classify the image type as ONE of: "document_page", "photograph", "chart", "figure", "diagram", "table", "form", "title_page", "cover_page", "advertisement", "other".

5. ENTITIES: List any notable entities, objects, or items visible. For document pages: book titles, author names, publisher names, dates, ISBN. For photographs: people, places, objects. Be specific and extract exact values when possible.

Be precise, thorough, and base all responses only on what is actually visible in the image. Do not speculate or infer information not present.
```

**Output Model:**

```python
class ImageVLMMetadata(BaseModel):
    description: str | None
    text_content: str | None
    language: str | None
    scene_classification: str | None
    entities: list[dict] = []
```

**Unified Model:** Same Gemma-4 model via vLLM handles both text semantic extraction and image vision understanding.

---

### 2.4 Text + Vision LLM with vLLM (Unified Engine)

**Changed from:** llama-cpp + GGUF model
**Changed to:** vLLM + PyTorch + HuggingFace model

**Benefits:**

- GPU-accelerated inference
- Automatic batching
- HuggingFace native support
- Better throughput on A100

**Implementation:**

```python
from vllm import LLM, SamplingParams

class ExtractionPipeline:
    def __init__(self):
        # vLLM loads model from HuggingFace
        self.llm = LLM(
            model="google/gemma-4-E4B-it",
            tensor_parallel_size=1,  # Use 1 A100
            dtype="bfloat16",
            max_model_len=8192,
        )

    def extract_semantic_fields(self, text: str) -> dict[str, Any]:
        prompt = f"""You are an expert book metadata extractor...
        OCR Text:
        {text[:3000]}
        JSON:
        """

        sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
        outputs = self.llm.generate([prompt], sampling_params)
        # Parse output...
```

**Environment Variables:**

| Variable                    | Description                                |
| --------------------------- | ------------------------------------------ |
| `VLLM_MODEL`                | Gemma-4 model ID (`google/gemma-4-E4B-it`) |
| `HF_TOKEN`                  | HuggingFace access token (required)        |
| `VLLM_TENSOR_PARALLEL_SIZE` | GPU count (default: 1)                     |

---

### 2.5 CLI Mode Update (Unified vLLM Engine)

**Status:** Code update required to match documentation

**Current State (Code):**

- `pipeline.py` uses `llama_cpp.Llama` for inference
- `main.py` CLI routes to `ExtractionPipeline` which uses llama-cpp
- No GPU acceleration, no vision support for images

**Target State (Documentation):**

- CLI uses same `ExtractionPipeline` as API
- Unified vLLM engine handles both text semantic extraction AND image vision
- Same environment variables: `VLLM_MODEL`, `HF_TOKEN`, `VLLM_TENSOR_PARALLEL_SIZE`

**CLI Commands:**

```bash
# Text file extraction - uses Gemma-4 vLLM
metaextractor book.pdf output.json

# Image extraction - uses Gemma-4 vLLM for description (Level 2)
metaextractor cover.jpg output.json

# JSON/MD file extraction - uses Gemma-4 vLLM
metaextractor metadata.json output.json

# With benchmark mode
metaextractor book.pdf output.json --benchmark

# Start API server (uses same vLLM engine)
metaextractor --api
```

**Code Changes Required:**

1. Replace `llama_cpp` import with `vllm`:

```python
# Before
from llama_cpp import Llama

# After
from vllm import LLM, SamplingParams
```

2. Update `ExtractionPipeline.__init__`:

```python
# Before
self.llm = Llama(model_path=effective_model_path, n_ctx=8192, verbose=False)

# After
self.llm = LLM(
    model=os.getenv("VLLM_MODEL", "google/gemma-4-E4B-it"),
    tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
    dtype="bfloat16",
    max_model_len=8192,
)
```

3. Update `extract_semantic_fields` method:

````python
# Before
output = self.llm(prompt, max_tokens=256, stop=["```"], echo=False, stream=False)
text_out = output["choices"][0]["text"].strip()

# After
sampling_params = SamplingParams(temperature=0.7, max_tokens=256, stop=["```"])
outputs = self.llm.generate([prompt], sampling_params)
text_out = outputs[0].outputs[0].text.strip()
````

4. Remove `_download_file_if_missing` and GGUF download logic (vLLM handles HuggingFace downloads natively)

5. Add vision support to `process_image`:

```python
async def process_image(self, image_path: str, benchmark: bool = False) -> dict[str, Any]:
    # EXIF extraction
    metadata_dict = extract_image_metadata(image_path)
    img_meta = ImageMetadata(**metadata_dict)

    # VLM description via Gemma-4 vLLM
    vlm_description = await self.vlm_client.describe_image(image_path)

    # Combine results
    result = ExtractionResult(
        image_metadata=img_meta,
        image_vlm_metadata=vlm_description
    )
    ...
```

6. Create `vlm_client.py` for image vision:

```python
from vllm import LLM, SamplingParams

class VLMClient:
    def __init__(self):
        self.llm = LLM(
            model=os.getenv("VLLM_MODEL", "google/gemma-4-E4B-it"),
            max_model_len=8192,
        )

    async def describe_image(self, image_path: str) -> dict:
        # Encode image and send to Gemma-4 vLLM
        ...
```

**Environment Variables for CLI:**

| Variable                    | Description              | Default                 |
| --------------------------- | ------------------------ | ----------------------- |
| `VLLM_MODEL`                | Model ID on HuggingFace  | `google/gemma-4-E4B-it` |
| `HF_TOKEN`                  | HuggingFace access token | (required)              |
| `VLLM_TENSOR_PARALLEL_SIZE` | GPU count                | `1`                     |

**Verification:**

```bash
# Test CLI mode
metaextractor test.pdf output.json --benchmark

# Check vLLM is loaded (should see GPU memory allocation)
nvidia-smi

# Test image Level 2 extraction
metaextractor test_image.jpg output.json
```

---

### 2.6 OCR Output Format Standardization

```json
{
  "transcription": "Full extracted text content...",
  "confidence": 0.95,
  "language": "en",
  "extraction_type": "ocr",
  "quality_score": 0.92,
  "notes": "Multi-page document processed",
  "segments": [
    {
      "start": 0,
      "end": 150,
      "text": "Page 1 content...",
      "confidence": 0.96,
      "proofread": false,
      "bbox": [x1, y1, x2, y2],
      "type": "paragraph",
      "reading_order": 1
    }
  ],
  "summary": "Document about...",
  "named_entities": [
    {
      "text": "ISBN",
      "type": "identifier",
      "value": "978-0-123456-78-9"
    }
  ],
  "model_name": "vparse-pipeline",
  "processing_date": "2026-04-30T12:00:00Z",
  "metadata": {
    "page_count": 42,
    "language_list": ["en"]
  }
}
```

---

### 2.6 Celery Foundation (Async-Ready)

**New Module:** `metaextractor/tasks.py`

```python
from celery import Celery

celery_app = Celery("metaextractor")
celery_app.config_from_object("metaextractor.celery_config")

@celery_app.task(bind=True, name="metaextractor.extract")
def extract_task(self, file_path: str, options: dict):
    """Celery task for async extraction - currently sync"""
    result = run_extraction_sync(file_path, options)
    return result
```

**Configuration:** `metaextractor/celery_config.py`

```python
task_always_eager = True  # True = sync, False = async (Phase 3)
```

**New API Endpoints:**

```python
@app.post("/extract/async")
async def extract_async(file: UploadFile = File(...)):
    task = extract_task.delay(temp_path, {"lang": lang.value})
    return {"job_id": task.id, "status": "submitted"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    task = celery_app.AsyncResult(job_id)
    return {"job_id": job_id, "status": task.state}
```

---

### Phase 2 Summary

| Feature                  | Module                         | Status     |
| ------------------------ | ------------------------------ | ---------- |
| vLLM Engine              | `vlm_client.py`, `pipeline.py` | ✅ Complete |
| Hardware Detection       | `hardware.py`                  | ✅ Complete |
| Model Registry           | `models_registry.py`           | ✅ Complete |
| Model CLI Commands       | `main.py`                      | ✅ Complete |
| Magazine Extraction      | `pipeline.py`                  | ✅ Complete |
| Model Downloader         | `scripts/setup_models.py`      | ✅ Complete |
| Docker GPU Support       | `docker-compose.yml`           | ✅ Complete |
| Audio Metadata           | `media_utils.py`               | 📋 Deferred |
| Video Metadata           | `media_utils.py`               | 📋 Deferred |
| Pre-OCR Regex            | `pdf_metadata_extractor.py`    | 📋 Deferred |
| Image VLM                | `vlm_client.py`                | 📋 Deferred |
| OCR Format Standard      | `models.py`, `pipeline.py`     | 📋 Deferred |

---

## Phase 3: Async Processing & Distributed Workers ✅ COMPLETE

### Objectives

- Activate Celery queue for true async processing
- Implement GPU worker pool for LLM processing
- Implement CPU worker pool for image extraction
- Add async API endpoints

### Completed

| Feature | Module | Status |
|---------|--------|--------|
| Redis Broker | `celery_config.py` | ✅ |
| Celery Tasks | `tasks.py` | ✅ |
| Queue Routing | `celery_config.py` | ✅ |
| GPU Worker | `docker-compose.yml` | ✅ |
| CPU Worker | `docker-compose.yml` | ✅ |
| Async API | `main.py` | ✅ |
| Model Downloader Service | `docker-compose.yml` | ✅ |

### Queue Configuration

| Queue       | Worker Type | Concurrency | Purpose                         |
| ----------- | ----------- | ----------- | ------------------------------- |
| `vlm_queue` | GPU         | 2           | LLM inference (PDF/text)        |
| `default_queue` | CPU     | 8           | Image EXIF extraction           |

### Worker Launch

```bash
# GPU worker for LLM tasks
celery -A metaextractor.tasks worker \
    --hostname=vlm-worker@%h \
    --concurrency=2 \
    -Q vlm_queue

# CPU worker for image extraction
celery -A metaextractor.tasks worker \
    --hostname=extraction-worker@%h \
    --concurrency=8 \
    -Q default_queue
```

---

## Phase 4: Scale Optimization

### Objectives

- VLM batching for GPU efficiency
- Redis caching layer
- Kubernetes deployment
- Rate limiting and throttling

### 4.1 VLM Batching

```python
class VLMProcessor:
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size

    async def process_batch(self, image_paths: list[str]) -> list[dict]:
        """Batch process images through Gemma-4 vLLM"""
        # Process in batches for GPU efficiency
```

### 4.2 Redis Caching

```python
@cache_result(expire=3600)
async def extract_with_cache(file_path: str, options: dict):
    """Cached extraction"""
    return await run_extraction(file_path, options)
```

---

## Module Architecture

### Deep Modules (Testable in Isolation)

| Module                      | Responsibility                   | Dependencies        |
| --------------------------- | -------------------------------- | ------------------- |
| `vlm_client.py`             | vLLM singleton wrapper           | vllm, hardware      |
| `validation.py`             | ISBN validation                  | regex               |
| `image_utils.py`            | EXIF extraction                  | PIL, piexif         |
| `pipeline.py`               | Orchestration + LLM inference    | vllm, vparse_client |
| `hardware.py`               | GPU/CPU/TPU detection            | pynvml, torch       |
| `models_registry.py`        | Model cache scanner              | huggingface_hub     |

### Shallow Modules (Adapters)

| Module             | Responsibility          |
| ------------------ | ----------------------- |
| `vparse_client.py` | VParse API adapter      |
| `external_api.py`  | OpenLibrary API adapter |

---

## Testing Strategy

### Unit Tests

- Mock external APIs (VParse, OpenLibrary)
- Test regex patterns with known inputs
- Validate data model serialization
- Test GPS normalization edge cases
- Mock vLLM LLM for semantic extraction tests

### Integration Tests

- Real VParse API calls
- End-to-end pipeline tests
- Celery task execution tests
- Async API endpoint tests

### Coverage

- Target: 90%+ coverage
- Current: 95% (140 tests)

---

## Deployment Checklist

### Phase 2

- [x] Deploy VParse OCR backend
- [x] Configure `VPARSE_API_URL`
- [x] Install vLLM and PyTorch with CUDA support
- [x] Configure HuggingFace cache directory
- [x] Test PDF extraction
- [x] Test image EXIF pipeline
- [x] Verify hardware detection
- [x] Update API documentation

### Phase 3

- [x] Deploy Redis
- [x] Configure Celery broker/result backend
- [x] Launch GPU workers
- [x] Launch CPU workers
- [x] Enable async queue
- [x] Add async API endpoints

### Phase 4

- [ ] Configure Kubernetes/Helm
- [ ] Set up auto-scaling
- [ ] Implement caching layer
- [ ] Load testing

---

## Glossary

| Term                   | Definition                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| **vLLM**               | High-throughput LLM inference engine with GPU acceleration                                        |
| **VLM**                | Vision Language Model - AI model that understands both images and text                            |
| **Celery**             | Async task queue for Python                                                                       |
| **VParse**             | OCR engine based on mineru-dots                                                                   |
| **HF_HOME**            | HuggingFace cache directory environment variable                                                  |

---

## References

- VParse (mineru-dots): https://github.com/suryamanoj4/mineru-dots
- vLLM Documentation: https://docs.vllm.ai/
- OpenLibrary API: https://openlibrary.org/developers/api
- Celery Documentation: https://docs.celeryproject.org/
- HuggingFace Hub: https://huggingface.co/docs/hub/
