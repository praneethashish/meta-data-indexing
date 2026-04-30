# Phase 4: Scale Optimization

**Status:** 📋 FUTURE

## Overview

Phase 4 focuses on optimizing the pipeline for large-scale processing (130k+ files), including VLM batching, distributed caching, and Kubernetes deployment.

---

## Objectives

1. **VLM batching** for GPU efficiency
2. **Redis caching layer** for repeated extractions
3. **Kubernetes deployment** for auto-scaling
4. **Rate limiting** and throttling
5. **Performance monitoring**

---

## 4.1 VLM Batching

### Problem

Processing images one-by-one through Gemma-4 vLLM is GPU-inefficient. Small batch sizes underutilize GPU memory; large batches cause OOM.

### Solution

```python
class VLMProcessor:
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size

    async def process_batch(self, image_paths: list[str]) -> list[dict]:
        """
        Batch process images through Gemma-4 vLLM
        Optimizes GPU utilization while avoiding OOM
        """
        results = []
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]

            # Process batch through vLLM
            batch_results = await self.process_vllm_batch(batch)
            results.extend(batch_results)

            # Clear GPU cache after each batch
            torch.cuda.empty_cache()

        return results

    async def process_vllm_batch(self, image_paths: list[str]) -> list[dict]:
        """Process a single batch through Gemma-4 vLLM"""
        # Prepare inputs
        images = [Image.open(p) for p in image_paths]
        prompts = [self.vlm_prompt] * len(images)

        # Batch inference via vLLM
        outputs = self.llm.generate(prompts, images=images)

        # Decode outputs
        results = []
        for output in outputs:
            decoded = self.processor.decode(output, skip_special_tokens=True)
            results.append(self.parse_vlm_output(decoded))

        return results
```

### Batch Size Guidelines

| GPU VRAM | Batch Size | Images/Second |
|----------|------------|---------------|
| A100 80GB | 8 | ~2-3 |
| A100 40GB | 4 | ~1.5-2 |
| V100 32GB | 4 | ~1-1.5 |
| T4 16GB | 2 | ~0.5-1 |

---

## 4.2 Redis Caching

### Cache Strategy

```python
import hashlib
import json
from functools import wraps

def generate_cache_key(file_path: str, options: dict) -> str:
    """Generate unique cache key based on file hash and options"""
    file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
    options_hash = hashlib.md5(json.dumps(options, sort_keys=True).encode()).hexdigest()
    return f"extraction:{file_hash}:{options_hash}"

def cache_result(expire: int = 3600, namespace: str = "cache"):
    """Cache extraction results in Redis"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(args[0], kwargs)

            # Check cache
            cached = redis_client.get(f"{namespace}:{cache_key}")
            if cached:
                return json.loads(cached)

            # Execute and cache
            result = await func(*args, **kwargs)
            redis_client.setex(f"{namespace}:{cache_key}", expire, json.dumps(result))
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expire=3600, namespace="extraction")
async def extract_with_cache(file_path: str, options: dict):
    """Cached extraction"""
    return await run_extraction(file_path, options)
```

### Cache Invalidation

| Event | Action |
|-------|--------|
| File modified | Invalidate cache for that file |
| Model update | Flush all cache |
| Manual flush | Admin API endpoint |

---

## 4.3 Kubernetes Deployment

### API Deployment

```yaml
# kubernetes/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bookextractor-api
  labels:
    app: bookextractor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bookextractor
  template:
    metadata:
      labels:
        app: bookextractor
    spec:
      containers:
      - name: api
        image: bookextractor:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
        env:
        - name: CELERY_BROKER_URL
          valueFrom:
            secretKeyRef:
              name: bookextractor-secrets
              key: celery-broker-url
        - name: VPARSE_API_URL
          value: "http://bookextractor-vparse:8000"
---
apiVersion: v1
kind: Service
metadata:
  name: bookextractor-api
spec:
  selector:
    app: bookextractor
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### VLM Worker Deployment (GPU)

```yaml
# kubernetes/vlm-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bookextractor-vlm-worker
  labels:
    app: bookextractor-vlm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: bookextractor-vlm
  template:
    metadata:
      labels:
        app: bookextractor-vlm
    spec:
      containers:
      - name: worker
        image: bookextractor:latest
        command: ["celery", "-A", "bookextractor.tasks", "worker", "-Q", "vlm_queue"]
        resources:
          limits:
            memory: "16Gi"
            nvidia.com/gpu: 1  # A100
          requests:
            memory: "8Gi"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
---
apiVersion: v1
kind: Service
metadata:
  name: bookextractor-vparse
spec:
  selector:
    app: bookextractor-vparse
  ports:
  - port: 8000
    targetPort: 8000
```

### HPA (Horizontal Pod Autoscaler)

```yaml
# kubernetes/api-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bookextractor-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bookextractor-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: External
    external:
      metric:
        name: redis_connected_clients
      target:
        type: AverageValue
        averageValue: 100
```

---

## 4.4 Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/extract")
@limiter.limit("100/minute")
async def extract_rate_limited(request: Request, file: UploadFile = File(...)):
    """Rate-limited extraction endpoint"""
    pass

@app.post("/extract/batch")
@limiter.limit("10/minute")
async def batch_extract_rate_limited(request: Request, files: list[UploadFile] = File(...)):
    """Rate-limited batch endpoint"""
    pass
```

---

## 4.5 Performance Monitoring

### Metrics to Track

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Queue depth | Redis | > 1000 |
| Task failure rate | Celery | > 5% |
| API latency p95 | FastAPI | > 10s |
| GPU utilization | nvidia-smi | < 30% |
| Worker memory | Kubernetes | > 80% |

### Dashboard Panels

1. **Overview**
   - Total extractions (24h)
   - Success/failure rate
   - Average processing time

2. **Queues**
   - Queue depth by queue
   - Tasks processing vs queued
   - Worker health

3. **GPU**
   - Utilization %
   - Memory usage
   - Batch throughput

4. **API**
   - Request rate
   - Response time histogram
   - Error rate

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Throughput (CPU) | 1000+ docs/hour |
| Throughput (GPU VLM) | 200+ images/hour |
| API p95 latency | < 5s |
| API p99 latency | < 10s |
| Cache hit rate | > 60% |
| Worker utilization | > 70% |

---

## Out of Scope

- Multi-tenant isolation (Phase N)
- Custom VLM prompts per tenant (Phase N)
- RAG pipeline integration (Phase N)

---

## Next Steps

➡️ **[Phase N: Future Enhancements](Phase-N-Future-Enhancements)**

---

*Status: Planned for future implementation*
