# Phase 2: Multimedia & VLM Enhancement

**Status:** 🚧 IN PROGRESS

## Overview

Phase 2 extends the pipeline with audio/video metadata extraction, image Level 2 VLM enhancement using Gemma-4 via vLLM, pre-OCR regex extraction, standardized OCR output format, and Celery foundation for async processing.

---

## Features

### Completed Features

| Feature | Status | Module |
|---------|--------|--------|
| Audio Metadata | 📋 TODO | `media_utils.py` |
| Video Metadata | 📋 TODO | `media_utils.py` |
| Pre-OCR Regex | 📋 TODO | `pdf_metadata_extractor.py` |
| Image VLM (Gemma-4) | 📋 TODO | `vlm_client.py` |
| OCR Format Standard | 📋 TODO | `models.py`, `pipeline.py` |
| Celery Foundation | 📋 TODO | `tasks.py`, `celery_config.py` |
| API Endpoints | 📋 TODO | `main.py` |

---

## Architecture

### Extended Format Router

```
                    ┌─────────────────────────────────────┐
                    │           /extract POST             │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │        Format Router              │
                    │      (by extension)                │
                    └─────────────────┬───────────────────┘
        ┌───────────┬────────┬────────┬────────┬────────┬────────┐
        │           │        │        │        │        │        │        │
   ┌────▼────┐ ┌────▼────┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌───▼───┐ ┌───▼───┐
   │   PDF   │ │  Text   │ │Image│ │Audio│ │Video│ │  MD   │ │  JSON │
   └────┬────┘ └────┬────┘ └──┬──┘ └──┬──┘ └──┬──┘ └───┬───┘ └───┬───┘
        │           │         │        │        │        │         │
        ▼           │         │        │        │        │         │
   ┌─────────┐     │         │        │        │        │         │
   │Pre-OCR  │     │         │        │        │        │         │
   │ Regex   │     │         │        │        │        │         │
   └────┬────┘     │         │        │        │        │         │
        │          │         │        │        │        │         │
        ▼          ▼         ▼        │        │        ▼         ▼
   ┌─────────┐ ┌─────────┐ ┌─────┐    │        │    ┌─────────┐
   │ vParse  │ │ Gemma-4│ │EXIF │    │        │    │ Gemma-4 │
   │   OCR   │ │  vLLM  │ │+VLM │    │        │    │  vLLM   │
   └────┬────┘ └─────────┘ └──┬──┘    │        │    └─────────┘
        │                     │       │        │
        │                     │       │        │
        └──────────┬──────────┴───────┴────────┘
                   │
              ┌────▼────┐
              │OpenLib  │
              │ISBN     │
              └────┬────┘
                   │
              ┌────▼──────────┐
              │  JSON Output  │
              └───────────────┘
        │                     │       │        │
        │                     │       │        │
        └──────────┬──────────┴───────┴────────┘
                   │
              ┌────▼────┐
              │OpenLib  │
              │ISBN     │
              └────┬────┘
                   │
              ┌────▼──────────┐
              │  JSON Output  │
              └───────────────┘
```

---

## Feature Specifications

### 2.1 Audio/Video Metadata Extraction

**New Module:** `bookextractor/media_utils.py`

#### Audio Metadata (.mp3, .wav, .m4a)

```python
class AudioMetadata(BaseModel):
    duration: float | None          # seconds
    bitrate: int | None             # bits per second
    sample_rate: int | None         # Hz
    channels: int | None            # 1=mono, 2=stereo
    codec: str | None               # e.g., "mp3", "aac"
    format: str | None              # e.g., "MP3", "WAV"
    container: str | None           # e.g., "ID3", "RIFF"
    size: int | None                # bytes
    warnings: list[str] = []        # Multi-stream warnings
```

#### Video Metadata (.mp4, .mkv)

```python
class VideoMetadata(BaseModel):
    duration: float | None          # seconds
    width: int | None                # pixels
    height: int | None              # pixels
    codec: str | None               # e.g., "h264", "hevc"
    framerate: float | None         # fps
    bitrate: int | None             # bits per second
    format: str | None              # e.g., "MP4", "MKV"
    size: int | None                # bytes
```

#### ffprobe Command

```bash
ffprobe -v quiet -print_format json -show_format -show_streams <file>
```

---

### 2.2 Pre-OCR Regex Extraction

**New Module:** `bookextractor/pdf_metadata_extractor.py`

```python
def extract_pdf_metadata_pre_ocr(file_path: str) -> dict[str, Any]:
    """
    Quick metadata extraction before expensive OCR.
    Uses PyPDF2/pdfplumber for text extraction + regex.

    Returns:
        {
            "isbn": str | None,
            "title": str | None,
            "author": str | None,
            "publisher": str | None,
            "published_date": str | None,
            "is_digital": bool  # True if text extractable without OCR
        }
    """
```

#### Regex Patterns

| Field | Pattern |
|-------|---------|
| ISBN-10/13 | `(?:\bISBN(?:-1[03])?:?\s*)?([0-9Xx\-\s]{10,20})` |
| Title | `<h1[^>]*>(.*?)</h1>` or metadata fields |
| Author | `by\s+(.+?)(?:\n\|,\|$)` or metadata fields |
| Publisher | `published\s+by\s+(.+?)(?:\n\|,\|$)` or metadata fields |

#### Merge Priority

```
Pre-OCR Regex > OpenLibrary > LLM
     (highest)      (if ISBN)   (fallback)
```

---

### 2.3 Image Level 2 VLM (Gemma-4 via vLLM)

**Module:** `bookextractor/vlm_client.py` (or integrated into `pipeline.py`)

#### VLM Prompt for Gemma-4

```
Analyze this image extracted from a document and provide a structured response with the following:

1. DESCRIPTION: A detailed, precise description of what this image shows. If it's a document page, describe the layout, headings, and key content visible. If it's a photograph or figure, describe the scene, objects, and context.

2. TEXT_CONTENT: Extract ALL text visible in the image, preserving the reading order. Include any titles, captions, labels, or written content. If no text is present, state "No text detected".

3. LANGUAGE: Identify the primary language of any text detected (e.g., "English", "Telugu", "Hindi", "Mixed"). If no text, state "Not applicable".

4. SCENE_CLASSIFICATION: Classify the image type as ONE of: "document_page", "photograph", "chart", "figure", "diagram", "table", "form", "title_page", "cover_page", "advertisement", "other".

5. ENTITIES: List any notable entities, objects, or items visible. For document pages: book titles, author names, publisher names, dates, ISBN. For photographs: people, places, objects. Be specific and extract exact values when possible.

Be precise, thorough, and base all responses only on what is actually visible in the image. Do not speculate or infer information not present.
```

#### VLM Output Model

```python
class ImageVLMMetadata(BaseModel):
    description: str | None
    text_content: str | None
    language: str | None
    scene_classification: str | None
    entities: list[dict] = []

class ImageMetadata(BaseModel):
    # Existing EXIF fields...
    width: int
    height: int
    format: str
    # ... existing fields ...

    # New VLM fields
    vlm: ImageVLMMetadata | None = None
```

#### Gemma-4 VLM Integration

```
bookextractor → vLLM (Gemma-4) → Image description
                     ↑
                     │
              Image file + prompt
              via vLLM multimodal
```

---

### 2.4 OCR Output Format (PDFs)

#### Standard OCR Output Schema

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

#### Data Models

```python
class OCRSegment(BaseModel):
    start: int
    end: int
    text: str
    confidence: float
    proofread: bool = False
    bbox: list[float] | None = None
    type: str  # paragraph, heading, table, etc.
    reading_order: int

class NamedEntity(BaseModel):
    text: str
    type: str  # person, organization, date, identifier
    value: str | None = None

class OCRResult(BaseModel):
    transcription: str
    confidence: float
    language: str
    extraction_type: str = "ocr"
    quality_score: float
    notes: str | None = None
    segments: list[OCRSegment] = []
    summary: str | None = None
    named_entities: list[NamedEntity] = []
    model_name: str
    processing_date: str
    metadata: dict = {}
```

---

### 2.5 Celery Foundation (Async-Ready)

**New Files:**
- `bookextractor/tasks.py` - Task definitions
- `bookextractor/celery_config.py` - Configuration

#### Task Definitions

```python
@celery_app.task(bind=True, name="bookextractor.extract")
def extract_task(self, file_path: str, options: dict):
    """Celery task for async extraction - currently sync"""
    result = run_extraction_sync(file_path, options)
    return result

@celery_app.task(bind=True, name="bookextractor.batch_extract")
def batch_extract_task(self, file_paths: list[str], options: dict):
    """Batch extraction task"""
    results = []
    for i, path in enumerate(file_paths):
        self.update_state(state="PROGRESS", meta={"current": i + 1, "total": len(file_paths)})
        results.append(run_extraction_sync(path, options))
    return results
```

#### Configuration

```python
# celery_config.py
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
task_always_eager = True  # True = sync, False = async (Phase 3)
```

#### New API Endpoints

```python
@app.post("/extract/async")
async def extract_async(file: UploadFile, lang: OCRLanguage):
    """Submit extraction job, return job ID"""
    task = extract_task.delay(temp_path, {"lang": lang.value})
    return {"job_id": task.id, "status": "submitted"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result"""
    task = celery_app.AsyncResult(job_id)
    return {"job_id": job_id, "status": task.state, "result": task.result if task.ready() else None}
```

---

## New Modules Summary

| Module | Purpose | Status |
|--------|---------|--------|
| `media_utils.py` | Audio/video metadata via ffprobe | TODO |
| `pdf_metadata_extractor.py` | Pre-OCR regex extraction | TODO |
| `vlm_client.py` | Gemma-4 vLLM integration (text + vision) | TODO |
| `tasks.py` | Celery task definitions | TODO |
| `celery_config.py` | Celery configuration | TODO |

---

## Modified Modules

| Module | Changes |
|--------|---------|
| `models.py` | Add AudioMetadata, VideoMetadata, ImageVLMMetadata, OCRResult, OCRSegment, NamedEntity |
| `pipeline.py` | Integrate pre-OCR, VLM, OCR format standardization |
| `main.py` | New extensions, new async endpoints |

---

## Out of Scope

- ❌ Async queue activation (Phase 3)
- ❌ Distributed workers (Phase 3)
- ❌ VLM batching optimization (Phase 4)

---

## Next Steps

➡️ **[Phase 3: Async Processing](Phase-3-Async-Processing)**

---

*Status: Implementation in progress*
