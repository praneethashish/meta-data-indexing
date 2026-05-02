# Phase 1: Foundation

**Status:** ✅ COMPLETE

## Overview

Phase 1 established the foundation for the metadata extraction pipeline, implementing the core format routing, PDF/text/image processing pipelines, and Docker containerization.

---

## Implemented Features

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

---

## Architecture

### Format Router Flow

```
                     ┌─────────────────┐
                     │  /extract POST  │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │ Format Router   │
                     │ (by extension)  │
                     └────────┬────────┘
          ┌─────────────┬────┴────┬─────────────┐
          │             │         │             │
     ┌────▼───┐    ┌────▼───┐    │       ┌────▼───┐
     │  PDF   │    │  Text  │    │       │ Image  │
     │.md/.json    │.pdf    │    │       │.jpg/.png    │
     └────┬───┘    └────┬───┘    │       └────┬───┘
          │             │         │             │
          ▼             ▼         │             ▼
     ┌─────────┐  ┌─────────┐    │      ┌─────────┐
     │ Gemma-4 │  │ vParse  │    │      │ EXIF/   │
     │ vLLM    │  │   OCR   │    │      │  PIL    │
     └────┬────┘  └────┬────┘    │      └─────────┘
          │             │         │
          └──────┬──────┴─────────┘
                 │
            ┌────▼────┐
            │OpenLib   │
            │ISBN      │
            └────┬────┘
                 │
            ┌────▼────────┐
            │ JSON Output  │
            └─────────────┘
```

---

## Components

### Source Files

| Module             | Path                             | Responsibility                     |
| ------------------ | -------------------------------- | ---------------------------------- |
| `main.py`          | `bookextractor/main.py`          | FastAPI application, format router |
| `pipeline.py`      | `bookextractor/pipeline.py`      | Extraction pipeline orchestrator   |
| `vparse_client.py` | `bookextractor/vparse_client.py` | VParse API client                  |
| `image_utils.py`   | `bookextractor/image_utils.py`   | EXIF extraction utilities          |
| `models.py`        | `bookextractor/models.py`        | Pydantic data models               |
| `validation.py`    | `bookextractor/validation.py`    | ISBN validation                    |
| `external_api.py`  | `bookextractor/external_api.py`  | OpenLibrary API client             |

### Data Models

```python
class BookMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    publisher: str | None = None
    isbn: str | None = None
    published_date: str | None = None
    confidence: ConfidenceScores

class ImageMetadata(BaseModel):
    width: int
    height: int
    format: str
    color_space: str | None = None
    bit_depth: int | None = None
    exif_camera_make: str | None = None
    exif_camera_model: str | None = None
    exif_date_taken: str | None = None
    exif_gps_latitude: float | None = None
    exif_gps_longitude: float | None = None
    exif_lens: str | None = None
    dpi_horizontal: float | None = None
    dpi_vertical: float | None = None

class ExtractionResult(BaseModel):
    book_metadata: BookMetadata | None = None
    image_metadata: ImageMetadata | None = None
```

---

## API Endpoints

### POST /extract

**Request:**

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@document.pdf" \
  -F "lang=en"
```

**Response:**

```json
{
  "book_metadata": {
    "title": "Example Book Title",
    "author": "John Doe",
    "publisher": "Example Publisher",
    "isbn": "978-0-123456-78-9",
    "published_date": "2024",
    "confidence": {
      "title": 0.95,
      "author": 0.88,
      "publisher": 0.75,
      "isbn": 1.0,
      "published_date": 0.7
    }
  }
}
```

### GET /health

**Response:**

```json
{ "status": "ok" }
```

---

## Processing Pipelines

### PDF Pipeline

1. Receive PDF file
2. Send to VParse OCR API (`/file_parse`)
3. Extract text from response (`content_list` or `md_content`)
4. Clean text (remove image references, normalize whitespace)
5. Extract ISBN via regex
6. Send text to Gemma-4 LLM for semantic extraction
7. Validate ISBN via OpenLibrary (if found)
8. Merge results (OpenLibrary > LLM)
9. Calculate confidence scores
10. Return JSON result

### Text Pipeline

1. Read `.md` or `.json` file content
2. Send to Gemma-4 LLM for semantic extraction
3. Extract ISBN via regex
4. Validate ISBN via OpenLibrary (if found)
5. Return JSON result

### Image Pipeline

1. Open image with PIL
2. Extract basic metadata (width, height, format, DPI, color space)
3. Extract EXIF data (camera make/model, date, lens, GPS)
4. Normalize GPS coordinates to decimal degrees
5. Return JSON result

---

## Docker Configuration

### Services

| Service         | Dockerfile                 | Description                         |
| --------------- | -------------------------- | ----------------------------------- |
| `bookextractor` | `Dockerfile.bookextractor` | FastAPI application with vLLM + GPU |
| `vparse`        | `Dockerfile.vparse`        | VParse OCR API (CPU-only)           |
| `vparse-lite`   | `Dockerfile.vparse.lite`   | Tesseract-only OCR (lightweight)    |
| `vparse-gpu`    | `Dockerfile.vparse.gpu`    | VParse with GPU support for OCR     |

### Launch

```bash
docker-compose up --build
```

---

## Environment Variables

| Variable                   | Description                        | Default                            |
| -------------------------- | ---------------------------------- | ---------------------------------- |
| `VPARSE_API_URL`           | VParse API endpoint                | `http://localhost:8000/file_parse` |
| `BOOKEXTRACTOR_MODELS_DIR` | Local models directory             | `./models`                         |
| `VLLM_MODEL`               | Gemma-4 model ID                   | `google/gemma-4-E4B-it`            |
| `HF_TOKEN`                 | HuggingFace token for model access | (required)                         |

---

## Testing

### Test Coverage

| Test File                     | Coverage                           |
| ----------------------------- | ---------------------------------- |
| `tests/test_pipeline.py`      | Pipeline processing logic          |
| `tests/test_vparse_client.py` | VParse API integration             |
| `tests/test_image_utils.py`   | EXIF extraction, GPS normalization |
| `tests/test_main.py`          | API endpoints                      |
| `tests/test_external_api.py`  | OpenLibrary integration            |

### Run Tests

```bash
pytest tests/ -v
```

---

## Out of Scope

- ❌ Audio/Video metadata extraction (Phase 2)
- ❌ Image Level 2 VLM description (Phase 2)
- ❌ Pre-OCR regex extraction (Phase 2)
- ❌ Async processing with Celery (Phase 3)
- ❌ Distributed workers (Phase 3)

---

## Next Steps

➡️ **[Phase 2: Multimedia & VLM Enhancement](Phase-2-Multimedia-VLM)**

---

_Status: Completed - All Phase 1 features implemented and tested_
