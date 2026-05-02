# Phase 3: Async Processing & Distributed Workers

**Status:** 📋 FUTURE

## Overview

Phase 3 activates the Celery async queue foundation from Phase 2, enabling distributed worker deployment, GPU worker pools for VLM processing, and batch processing capabilities.

---

## Objectives

1. **Activate Celery queue** for true async processing
2. **Implement GPU worker pool** for VLM (Gemma-4 via vLLM) processing
3. **Implement CPU worker pool** for standard extraction
4. **Add batch processing** with progress tracking
5. **Implement result persistence** to database

---

## Architecture

### Worker Deployment

```
                              ┌─────────────────┐
                              │   Redis Broker  │
                              │   (Task Queue)  │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
               ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
               │  API    │       │  GPU    │       │  CPU    │
               │ Service │       │ Worker  │       │ Worker  │
               └─────────┘       │(VLLM)  │       │(Extract)│
                               └────┬────┘       └────┬────┘
                                    │                  │
                              ┌────▼────┐       ┌────▼────┐
                              │ Gemma-4 │       │ vParse  │
                              │ (Text +  │       │  OCR    │
                              │  Vision) │       │  Lite   │
                              │  vLLM   │       └─────────┘
                              └─────────┘
```

                              ┌─────────────────┐
                              │   Redis Broker  │
                              │   (Task Queue)  │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
               ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
               │  API    │       │  GPU    │       │  CPU    │
               │ Service │       │ Worker  │       │ Worker  │
               └─────────┘       │(VLM)    │       │(Extract)│
                               └────┬────┘       └────┬────┘
                                    │                  │

┌────▼────┐ ┌────▼────┐
│Gemma-4 │ │ vParse │
│ vLLM │ │ OCR │
│(text+ │ │ │
│ vision) │ │ │
└─────────┘ └─────────┘

````

### Queue Configuration

| Queue | Worker Type | Concurrency | Purpose |
|-------|-------------|-------------|---------|
| `default` | CPU | 8 | Standard extraction (PDF, text, audio, video) |
| `vlm_queue` | GPU | 2 | Text LLM + Vision LLM (Gemma-4 vLLM unified) |
| `celery` | CPU | 4 | Internal Celery tasks |

---

## Implementation Details

### 3.1 Queue Activation

**Changes to `celery_config.py`:**

```python
# Phase 3: Activate async queue
task_always_eager = False  # Switch from sync to async
task_track_started = True
task_time_limit = 3600  # 1 hour max per task
task_soft_time_limit = 3000  # 50 min soft limit

# Worker configuration
worker_prefetch_multiplier = 1  # One task per worker at a time
worker_max_tasks_per_child = 100  # Restart worker after 100 tasks
````

### 3.2 Worker Launch Commands

```bash
# GPU worker for VLM tasks (A100)
celery -A bookextractor.tasks worker \
    --hostname=vlm-worker@%h \
    --concurrency=2 \
    --pool=prefork \
    -O fair \
    -Q vlm_queue \
    --max-memory-per-child=65536

# CPU worker for standard extraction
celery -A bookextractor.tasks worker \
    --hostname=extraction-worker@%h \
    --concurrency=8 \
    --pool=prefork \
    -Q default_queue,celery \
    --max-tasks-per-child=1000

# Batch worker for large jobs
celery -A bookextractor.tasks worker \
    --hostname=batch-worker@%h \
    --concurrency=2 \
    --pool=prefork \
    -Q batch_queue
```

### 3.3 Queue Routing

```python
@celery_app.task(bind=True, name="bookextractor.extract")
def extract_task(self, file_path: str, options: dict):
    """Route to appropriate queue based on file type"""
    ext = Path(file_path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        queue = "vlm_queue"  # GPU - Gemma-4 vLLM (text + vision unified)
    elif ext in PDF_EXTENSIONS:
        queue = "default_queue"  # CPU - vParse OCR
    else:
        queue = "default_queue"

    # Manual routing would require Celery router
    return run_extraction_sync(file_path, options)
```

### 3.4 Batch Processing API

```python
@app.post("/extract/batch")
async def extract_batch(
    files: list[UploadFile] = File(...),
    lang: OCRLanguage = Form(OCRLanguage.ENGLISH),
    priority: str = Form("normal")  # low, normal, high
):
    """Submit batch extraction job"""
    job = batch_extract_task.delay(
        [save_temp_file(f) for f in files],
        {"lang": lang.value, "priority": priority}
    )
    return {
        "batch_id": job.id,
        "file_count": len(files),
        "status": "submitted",
        "priority": priority
    }

@app.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get batch progress"""
    job = celery_app.AsyncResult(batch_id)
    return {
        "batch_id": batch_id,
        "status": job.state,
        "progress": job.info if hasattr(job, 'info') else None,
        "completed": job.ready(),
        "result": job.result if job.ready() else None
    }
```

### 3.5 Result Persistence

```python
# Store results in Redis with TTL
celery_app.conf.result_expires = 86400  # 24 hours

# Custom result backend with database persistence
@celery_app.task
def extract_with_persistence(self, file_path: str, options: dict):
    result = run_extraction_sync(file_path, options)

    # Persist to database
    save_extraction_result(
        job_id=self.request.id,
        file_path=file_path,
        result=result,
        status="completed",
        metadata={
            "file_size": os.path.getsize(file_path),
            "processing_time": time.time() - start_time
        }
    )

    return result
```

---

## Performance Targets

| Metric                | Target                 |
| --------------------- | ---------------------- |
| Documents/hour (CPU)  | 500-1000               |
| Images/hour (GPU VLM) | 100-200                |
| API Response (sync)   | < 5s for small files   |
| Batch job start       | < 30s after submission |
| Worker recovery       | < 10s after failure    |

---

## Monitoring

### Celery Flower

```bash
celery -A bookextractor.tasks flower --port=5555
```

### Metrics to Track

- Queue depth per worker type
- Task success/failure rate
- Processing time per file type
- Worker memory usage
- GPU utilization (VLM workers)

---

## Deployment

### Docker Compose (Phase 3 Update)

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    depends_on:
      - redis
      - vparse
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0

  worker-gpu:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: celery -A bookextractor.tasks worker -Q vlm_queue --hostname=vlm-worker@%h

  worker-cpu:
    build: .
    command: celery -A bookextractor.tasks worker -Q default_queue --hostname=extraction-worker@%h --concurrency=8
```

---

## Out of Scope

- ❌ Kubernetes deployment (Phase 4)
- ❌ VLM batching optimization (Phase 4)
- ❌ Redis caching layer (Phase 4)

---

## Next Steps

➡️ **[Phase 4: Scale Optimization](Phase-4-Scale-Optimization)**

---

_Status: Planned for future implementation_
