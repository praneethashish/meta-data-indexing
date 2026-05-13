# Implement vLLM with Gemma-4 for GPU-Accelerated Inference

## Status

**Type:** Implementation Task
**Priority:** High
**Phase:** Phase 2 - Multimedia & VLM Enhancement
**Status:** ✅ COMPLETE

## Summary

Replace `llama-cpp-python` with `vllm` for GPU-accelerated inference using `google/gemma-4-E4B-it`. The Gemma-4 model will be cached in the host's HuggingFace cache directory (`~/.cache/huggingface/hub/`) rather than the project `models/` volume.

## Motivation

### Current State

- Uses `llama-cpp-python` with quantized GGUF models
- CPU-bound inference with no GPU acceleration
- GGUF files stored in `models/` volume alongside VParse OCR models
- Model download handled via custom `_download_file_if_missing()` function

### Target State

- Uses `vllm` with native HuggingFace support
- GPU-accelerated inference via CUDA/PyTorch
- Model cached in host's `~/.cache/huggingface/hub/` (bind mount)
- `models/` volume reserved for VParse OCR models only

## Scope

### In Scope

- [x] Replace `llama-cpp` import with `vllm` in `pipeline.py`
- [x] Update `ExtractionPipeline.__init__` to use `vllm.LLM`
- [x] Update `extract_semantic_fields` to use vLLM's `SamplingParams` and `.generate()` API
- [x] Remove GGUF download logic (`_download_file_if_missing`, `resolve_model_paths`)
- [x] Update `pyproject.toml` dependencies
- [x] Update `.env.example` with new vLLM environment variables
- [x] Update `Dockerfile.bookextractor` with CUDA base image
- [x] Update `docker-compose.yml` with GPU resources and HF cache volume mount
- [x] Update `tests/test_pipeline.py` mocks for vLLM API
- [x] Update `scripts/setup_models.py` for HuggingFace download

### Not In Scope (Future Tasks)

- Image Level 2 VLM description via Gemma-4 vision
- Audio/Video metadata extraction
- Celery async processing

## Files to Modify

| File                          | Changes                                                              |
| ----------------------------- | -------------------------------------------------------------------- |
| `pyproject.toml`              | Replace `llama-cpp-python` with `vllm>=0.8.0`, add `huggingface_hub` |
| `bookextractor/pipeline.py`   | Replace LLM initialization and inference API                         |
| `bookextractor/models.py`     | Add `ImageVLMMetadata` placeholder                                   |
| `bookextractor/vlm_client.py` | Create placeholder for future image VLM                              |
| `.env.example`                | Replace GGUF env vars with `VLLM_MODEL`, `HF_TOKEN`                  |
| `Dockerfile.bookextractor`    | Use `nvidia/cuda` base image                                         |
| `docker-compose.yml`          | Add GPU resources, HF cache bind mount                               |
| `tests/test_pipeline.py`      | Update mocks for vLLM API                                            |
| `scripts/setup_models.py`     | Remove Gemma download                                                |

## Environment Variables

| Variable                    | Description                                          | Default                 |
| --------------------------- | ---------------------------------------------------- | ----------------------- |
| `VLLM_MODEL`                | HuggingFace model ID                                 | `google/gemma-4-E4B-it` |
| `HF_TOKEN`                  | HuggingFace access token (required for gated models) | -                       |
| `VLLM_TENSOR_PARALLEL_SIZE` | GPU count                                            | `1`                     |

## Storage Architecture

```
Host Filesystem:
~/.cache/huggingface/hub/    ← vLLM model cache (bind mount)
  └── models/
      └── google/
          └── gemma-4-E4B-it/  ← Downloaded automatically by vLLM

models volume (Docker):
/models/                      ← VParse OCR models only
```

## Docker Changes

### Dockerfile.bookextractor

- Base image: `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
- Install `uv` for fast dependency resolution
- Install `vllm>=0.8.0` via `uv pip install --system`
- Set `HF_HOME=/root/.cache/huggingface/hub`

### docker-compose.yml

```yaml
services:
  bookextractor:
    volumes:
      - ~/.cache/huggingface/hub:/root/.cache/huggingface/hub # HF cache
      - models:/models # VParse OCR
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - VLLM_MODEL=google/gemma-4-E4B-it
      - HF_TOKEN=${HF_TOKEN}
      - VLLM_TENSOR_PARALLEL_SIZE=1
```

## Code Changes

### pipeline.py - Import Change

```python
# Before
from llama_cpp import Llama

# After
from vllm import LLM, SamplingParams
```

### pipeline.py - Initialization

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

### pipeline.py - Inference

````python
# Before
output = self.llm(prompt, max_tokens=256, stop=["```"], echo=False, stream=False)
text_out = output["choices"][0]["text"].strip()

# After
sampling_params = SamplingParams(temperature=0.7, max_tokens=256, stop=["```"])
outputs = self.llm.generate([prompt], sampling_params)
text_out = outputs[0].outputs[0].text.strip()
````

## Verification

```bash
# Build
docker compose build bookextractor

# Run with GPU
docker compose up bookextractor

# Test extraction
curl -X POST http://localhost:8000/extract -F "file=@test.pdf"

# Verify HF cache on host
ls ~/.cache/huggingface/hub/

# Verify GPU usage
nvidia-smi
```

## Dependencies

- Docker with NVIDIA runtime (`nvidia-container-toolkit`)
- NVIDIA GPU with CUDA 12.4+ support
- HuggingFace account with accepted Gemma model terms

## Risks & Mitigations

| Risk                | Mitigation                                     |
| ------------------- | ---------------------------------------------- |
| GPU not available   | Fail fast with clear error message             |
| HF_TOKEN not set    | Require token for gated Gemma model            |
| Large image size    | Use `runtime` base (not `devel`), minimal deps |
| Model download slow | Pre-populate HF cache via model-downloader     |

## Labels

- `vllm`
- `gpu-acceleration`
- `phase-2`
- `gemma-4`
