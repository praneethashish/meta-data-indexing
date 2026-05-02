# Phase N: Future Enhancements

**Status:** 📋 FUTURE

## Overview

Phase N covers planned enhancements beyond the core pipeline, including streaming OCR, multi-tenant support, RAG pipeline integration, and specialized detectors.

---

## Priority Matrix

| Feature                   | Priority | Effort | Impact       |
| ------------------------- | -------- | ------ | ------------ |
| Streaming OCR             | Medium   | High   | Performance  |
| Custom VLM Prompts        | Medium   | Medium | Flexibility  |
| Multi-tenant Support      | Medium   | High   | Enterprise   |
| RAG Pipeline              | Medium   | High   | Use Cases    |
| Watermark Detection       | Low      | Medium | Quality      |
| Signature Detection       | Low      | Medium | Quality      |
| TOC Extraction            | Low      | Medium | Completeness |
| Cross-lingual Translation | Low      | High   | Reach        |

---

## N.1 Streaming OCR

### Description

Stream OCR results as pages are processed, rather than waiting for entire document.

### Use Case

- Very large documents (1000+ pages)
- Real-time display of OCR progress
- Memory-efficient processing

### Implementation

```python
async def process_pdf_streaming(pdf_path: str):
    """
    Stream OCR results page by page.
    Yields results as each page is processed.
    """
    pdf_reader = PdfReader(pdf_path)

    for page_num, page in enumerate(pdf_reader.pages):
        # Process single page
        page_text = await process_page(page)

        yield {
            "page": page_num,
            "total_pages": len(pdf_reader.pages),
            "text": page_text,
            "progress": (page_num + 1) / len(pdf_reader.pages)
        }

# API endpoint
@app.post("/extract/stream")
async def extract_streaming(file: UploadFile = File(...)):
    """Stream extraction results"""
    temp_path = save_temp_file(file)

    async def generate():
        async for result in process_pdf_streaming(temp_path):
            yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## N.2 Custom VLM Prompts

### Description

Allow users to define custom prompts for image description based on their specific needs.

### Use Cases

| Tenant Type | Custom Prompt                                          |
| ----------- | ------------------------------------------------------ |
| Library     | "Describe book covers, spines, and dust jackets"       |
| Archive     | "Identify photographs, maps, and historical documents" |
| Medical     | "Extract medical images, charts, and diagrams"         |
| Legal       | "Identify signatures, stamps, and official seals"      |

### Implementation

```python
class VLMConfiguration(BaseModel):
    prompt_template: str
    response_schema: dict | None = None
    temperature: float = 0.7
    max_tokens: int = 512

@app.post("/extract/image")
async def extract_image(
    file: UploadFile = File(...),
    vlm_config: VLMConfiguration | None = None
):
    """Extract with optional custom VLM configuration"""
    if vlm_config is None:
        vlm_config = DEFAULT_VLM_CONFIG

    result = await process_image_with_config(file, vlm_config)
    return result

# User-defined configuration
{
    "prompt_template": "Identify all official seals and stamps in this document: {image}",
    "response_schema": {
        "seals": ["string"],
        "stamps": ["string"],
        "authenticity": "boolean"
    }
}
```

---

## N.3 Multi-tenant Support

### Description

Isolate data and processing per tenant with quotas and billing.

### Features

- Tenant isolation (data, models, quotas)
- API key authentication
- Usage tracking and billing
- Tenant-specific VLM prompts

### Implementation

```python
class Tenant(BaseModel):
    id: str
    name: str
    api_key: str
    quotas: TenantQuotas
    vlm_config: VLMConfiguration | None = None

class TenantQuotas(BaseModel):
    requests_per_minute: int
    requests_per_day: int
    max_file_size_mb: int
    gpu_enabled: bool

@app.middleware
async def tenant_auth(request: Request, call_next):
    """Authenticate tenant and enforce quotas"""
    api_key = request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(401, "API key required")

    tenant = await get_tenant_by_api_key(api_key)
    if not tenant:
        raise HTTPException(401, "Invalid API key")

    # Check quota
    if not await check_quota(tenant):
        raise HTTPException(429, "Quota exceeded")

    request.state.tenant = tenant
    return await call_next(request)
```

---

## N.4 RAG Pipeline Integration

### Description

Chunk documents and index for retrieval-augmented generation use cases.

### Features

- Semantic chunking of documents
- Vector embedding generation
- Elasticsearch/Solr indexing
- Similarity search API

### Implementation

```python
class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: dict  # page, position, type
    embedding: list[float]

@app.post("/extract/index")
async def extract_and_index(file: UploadFile = File(...)):
    """Extract metadata and index for RAG"""
    # 1. Extract metadata
    result = await extract(file)

    # 2. Generate chunks
    chunks = await chunk_document(file, result)

    # 3. Generate embeddings
    embeddings = await generate_embeddings([c.content for c in chunks])

    # 4. Index in vector database
    await vector_db.index([
        {**chunk.dict(), "embedding": emb}
        for chunk, emb in zip(chunks, embeddings)
    ])

    return {
        "document_id": result.document_id,
        "chunk_count": len(chunks),
        "indexed": True
    }

@app.post("/search")
async def semantic_search(query: str, top_k: int = 10):
    """Search indexed documents"""
    # Generate query embedding
    query_embedding = await generate_embeddings([query])

    # Search
    results = await vector_db.search(
        query_embedding[0],
        top_k=top_k
    )

    return results
```

---

## N.5 Watermark Detection

### Description

Identify and extract watermark information from documents.

### Use Cases

- Copyright detection
- Document authenticity verification
- Source tracking

### Implementation

```python
async def detect_watermark(image_path: str) -> dict:
    """
    Detect watermarks in document images.
    Uses specialized model or rule-based detection.
    """
    # Implementation placeholder
    return {
        "has_watermark": bool,
        "watermark_text": str | None,
        "watermark_position": [x, y, w, h],
        "confidence": float
    }
```

---

## N.6 Signature Detection

### Description

Detect and extract signatures from documents.

### Use Cases

- Contract processing
- Document verification
- Automated approval workflows

### Implementation

```python
async def detect_signature(image_path: str) -> dict:
    """
    Detect signatures in document images.
    """
    return {
        "has_signature": bool,
        "signature_regions": [
            {"bbox": [x1, y1, x2, y2], "confidence": float}
        ],
        "extracted_signature_image": str | None  # Base64
    }
```

---

## N.7 Table of Contents Extraction

### Description

Extract structured Table of Contents from documents.

### Implementation

```python
async def extract_toc(pdf_path: str) -> dict:
    """
    Extract Table of Contents from PDF.
    Returns hierarchical structure.
    """
    return {
        "has_toc": bool,
        "entries": [
            {
                "level": 1,
                "title": "Chapter 1: Introduction",
                "page": 1,
                "children": [
                    {
                        "level": 2,
                        "title": "1.1 Background",
                        "page": 1
                    }
                ]
            }
        ]
    }
```

---

## N.8 Cross-lingual Metadata Translation

### Description

Translate metadata (title, author, publisher) between languages.

### Use Cases

- Multi-language document repositories
- Metadata normalization for search
- Localization support

### Implementation

```python
async def translate_metadata(
    metadata: BookMetadata,
    target_language: str = "en"
) -> BookMetadata:
    """
    Translate book metadata to target language.
    Uses translation LLM.
    """
    translated = await translate_text(
        f"Title: {metadata.title}\nAuthor: {metadata.author}\nPublisher: {metadata.publisher}",
        target_language=target_language
    )

    return metadata.copy(
        update={
            "title": translated.title,
            "author": translated.author,
            "publisher": translated.publisher,
            "original_language": detect_language(metadata.title)
        }
    )
```

---

## Backlog Items

| Item                 | Description                            | Blocking       |
| -------------------- | -------------------------------------- | -------------- |
| Custom OCR languages | Support additional PaddleOCR languages | None           |
| PDF/A extraction     | Extract from PDF/A archival format     | None           |
| Table to CSV         | Convert detected tables to CSV         | TOC extraction |
| Barcode detection    | Detect ISBN barcodes                   | None           |
| QR code extraction   | Extract URLs from QR codes             | None           |

---

_Status: Future enhancements - priority and timeline to be determined_
