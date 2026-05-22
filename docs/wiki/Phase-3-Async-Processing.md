# Phase 3: Async Processing & Distributed Workers

**Status:** ✅ COMPLETE

## Overview

Phase 3 activated the Celery async queue with Redis broker, implemented dual-queue worker architecture (GPU for LLM tasks, CPU for image tasks), and added async API endpoints.

---

## Completed Features

| Feature | Status | Module |
|---------|--------|--------|
| Redis Broker | ✅ | `celery_config.py`, `docker-compose.yml` |
| Celery Tasks | ✅ | `tasks.py` |
| Queue Routing | ✅ | `celery_config.py` |
| GPU Worker | ✅ | `docker-compose.yml` (worker-gpu) |
| CPU Worker | ✅ | `docker-compose.yml` (worker-cpu) |
| Async API | ✅ | `main.py` (`/extract/async`, `/jobs/{id}`) |
| Model Downloader Service | ✅ | `docker-compose.yml` (model-downloader) |
| Flower Monitoring | ✅ | `pyproject.toml` dependency |

---

## Architecture

### Queue Configuration

| Queue | Worker Type | Concurrency | Purpose |
|-------|-------------|-------------|---------|
| `vlm_queue` | GPU | 2 | PDF/text extraction (LLM inference) |
| `default_queue` | CPU | 8 | Image extraction (EXIF only) |

### Worker Configuration

```python
# celery_config.py
broker_url = redis://localhost:6379/0
result_backend = redis://localhost:6379/0
task_always_eager = False  # Async mode
task_track_started = True
task_time_limit = 3600  # 1 hour hard limit
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 100
result_expires = 86400  # 24 hours
```

### Task Routing

| Task | Queue | LLM |
|------|-------|-----|
| `extract_pdf` | `vlm_queue` | Yes |
| `extract_text` | `vlm_queue` | Yes |
| `extract_image` | `default_queue` | No |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract/async` | POST | Submit async job, returns `job_id` |
| `/jobs/{job_id}` | GET | Poll job status/result |

### Async Flow

```
POST /extract/async
  → Save file to uploads/{uuid}_{filename}
  → extract_pdf_task.delay(path, lang)  # or extract_image_task, extract_text_task
  → Redis queues to vlm_queue or default_queue
  → Worker picks up task
  → Result stored in Redis
  → Client polls GET /jobs/{job_id}
```

---

## Docker Services

| Service | Queue | GPU | Concurrency |
|---------|-------|-----|-------------|
| `metaextractor` | N/A (API) | Yes | N/A |
| `worker-gpu` | `vlm_queue` | Yes | 2 |
| `worker-cpu` | `default_queue` | No | 8 |
| `redis` | N/A | No | N/A |
| `model-downloader` | N/A | No | N/A |

---

## CLI Commands

```bash
# Start a Celery worker
uv run metaextractor worker --queue vlm_queue --concurrency 2
uv run metaextractor worker --queue default_queue --concurrency 8

# Start Flower monitoring
celery -A metaextractor.tasks flower --port=5555
```

---

## Next Steps

➡️ **[Phase 4: Scale Optimization](Phase-4-Scale-Optimization)**

---

_Last updated: Phase 3 complete_
