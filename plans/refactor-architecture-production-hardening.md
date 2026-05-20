# Refactor: Architectural Hardening for Production Readiness

## Problem Statement

The codebase was built as a prototype and has several architectural issues that make it fragile for production:

- **VLMClient singleton is not thread-safe**: Concurrent API requests can silently overwrite each other's LLM engine instance. No locking around initialization or access.
- **Celery tasks use `asyncio.run()`**: Creates a new event loop per task, blocking concurrent I/O and risking `RuntimeError` with async worker pools.
- **Upload file cleanup races with active jobs**: `@worker_ready` prunes files >= 24h old without checking in-flight tasks. A long VLM job could have its input deleted mid-processing.
- **Queue routing split across config and runtime**: `celery_config.py` routes `extract_pdf` to `vlm_queue` and `extract_image` to `default_queue`, but `main.py:extract_async()` overrides image routing dynamically at runtime. Two sources of truth.
- **`load_llm` flag conflates model loading semantics**: The flag means "load vLLM which might be used for vision" but reads as "load an LLM." Misleading.
- **`ExtractionPipeline` does too much**: One class handles PDF OCR orchestration, ISBN extraction, LLM prompting, content type detection, confidence scoring, JSON parsing, and text cleaning.
- **Hard-coded prompts inside methods**: Multi-line prompt strings embedded directly. Any prompt tweak requires a code change + redeploy.
- **No error boundaries**: `except Exception: return {}` swallows failures silently. Callers never know whether the LLM never loaded, the prompt failed, or the output was malformed.
- **No model readiness check in `/health`**: Returns OK even if vLLM is still downloading or crashed silently during init.
- **Dead code**: Unreachable `self.vlm_client is None` check inside a `try` block in `_extract_magazine_semantic_fields`.

## Solution

A phased refactoring that:

1. **Extracts responsibilities** from `ExtractionPipeline` into focused, testable modules (`prompts.py`, `content_detector.py`, `confidence_scorer.py`).
2. **Introduces an LLM abstraction layer** (`ModelManager`, `LLMBackend` interface) with thread-safe singleton management.
3. **Fixes the 7 specific concerns** (singleton safety, async Celery, cleanup race, queue routing, semantic naming, dead code, error boundaries).
4. **Improves observability** with a model-ready health check.
5. **Adds/updates tests** for every new and changed module.

All while keeping the public API (CLI, HTTP endpoints, Celery task signatures) backward-compatible.

## Commits

Each commit leaves the codebase in a working state with passing tests.

### Phase 1 — Pure extractions (no behavior change)

**Commit 1: Remove dead code in `_extract_magazine_semantic_fields`**
- Remove the redundant `self.vlm_client is None` check inside the `try` block.
- The outer method guard already handles this.

**Commit 2: Extract prompt strings to `prompts.py`**
- Create `bookextractor/prompts.py` with module-level constants:
  - `BOOK_EXTRACTION_PROMPT` (from `extract_semantic_fields`)
  - `MAGAZINE_EXTRACTION_PROMPT` (from `extract_magazine_semantic_fields`)
  - `IMAGE_ANALYSIS_PROMPT` (from `describe_image`)
- Import and use them in `pipeline.py` and `vlm_client.py`.
- No prompt content changes -- pure mechanical extraction.

**Commit 3: Extract `ContentDetector` from pipeline**
- Create `bookextractor/content_detector.py` with a `detect_content_type(text) -> str` function.
- Move all magazine pattern detection logic (Telugu + English patterns) into it.
- Remove `ExtractionPipeline._detect_content_type` and delegate to the new function.

**Commit 4: Extract `ConfidenceScorer` from pipeline**
- Create `bookextractor/confidence_scorer.py` with `calculate_book_confidence()` and `calculate_magazine_confidence()`.
- Move `calculate_confidence` out of `ExtractionPipeline`.
- The module handles both `ConfidenceScores` and `MagazineConfidenceScores` construction.

**Commit 5: Rename `load_llm` to `load_vlm` across the codebase**
- Rename the `__init__` parameter in `ExtractionPipeline`.
- Update all callers: `main.py` (sync endpoint, async endpoint, `get_vision_pipeline`), `tasks.py` (`get_pipeline`, `extract_pdf_task`, `extract_image_task`, `extract_text_task`).
- Clarifies that this flag controls VLM/multimodal capability, not text-only LLM.

### Phase 2 — New abstractions

**Commit 6: Create `LLMBackend` interface with text and vision implementations**
- Define `LLMBackend` ABC in a new `bookextractor/llm_backend.py`.
- Two implementations:
  - `TextLLMBackend`: for metadata extraction (book/magazine). Default `temperature=0.7`, `max_tokens=512`.
  - `VisionLLMBackend`: for image description. Default `temperature=0.2`, `max_tokens=512`, multimodal input.
- Each wraps `vLLM.generate()` with appropriate `SamplingParams` defaults.
- Additive -- no callers use the new interface yet.

**Commit 7: Create `ModelManager` for thread-safe singleton model lifecycle**
- Create `bookextractor/model_manager.py`.
- Wraps the vLLM `LLM` engine with `threading.Lock`-protected initialization.
- API:
  - `get_or_create(model_id, max_model_len, config) -> LLM`: thread-safe init.
  - `get_model() -> LLM | None`: non-blocking check.
  - `is_loaded() -> bool`: readiness check.
  - `unload()`: release resources.
- All access to the underlying LLM goes through the lock.

**Commit 8: Wire `ModelManager` into `VLMClient`**
- Replace `VLMClient._llm` class-level state with `ModelManager` instance.
- `VLMClient._initialize_llm` calls `ModelManager.get_or_create()`.
- `VLMClient.generate()` and `describe_image()` get the model through the manager.
- Keep `VLMClient`'s public API unchanged.
- Update test helpers to also clear ModelManager state between tests.

**Commit 9: Refactor `ExtractionPipeline` to use new modules**
- `process_pdf`, `process_image`, `process_text_file` delegate to `ContentDetector` and `ConfidenceScorer`.
- `extract_semantic_fields` and `extract_magazine_semantic_fields` use prompts from `prompts.py` and backends from `LLMBackend`.
- Pipeline becomes a thin orchestrator.

### Phase 3 — Behavioral fixes

**Commit 10: Unify queue routing in `celery_config.py`**
- Move all task-to-queue routing into `celery_config.py`'s `task_routes` dict.
- Remove the runtime queue override logic from `main.py:extract_async()`.
- The `use_vlm` flag on image tasks is handled via config, not runtime injection.

**Commit 11: Fix upload cleanup race with in-flight tracking**
- Add a module-level `_in_flight_files: set[str]` in `tasks.py`.
- Each Celery task registers its file path at start, unregisters in `finally` block.
- `cleanup_stale_uploads` skips files present in the in-flight set.
- Thread-safe via a lock.

**Commit 12: Add sync facade methods on `ExtractionPipeline`**
- Add `process_pdf_sync`, `process_image_sync`, `process_text_file_sync` wrappers.
- Each creates a new event loop, runs the async method, closes the loop.
- Celery tasks call the sync wrappers instead of `asyncio.run()` directly.
- The async originals remain for FastAPI direct use.

**Commit 13: Add typed error boundaries**
- Define `ModelNotAvailableError`, `PromptFailedError`, `ParsingError` in new `bookextractor/exceptions.py`.
- Replace bare `except Exception: return {}` in `extract_semantic_fields` and `_extract_magazine_semantic_fields` with specific catches.
- Pipeline methods propagate errors instead of silently returning null metadata.

### Phase 4 — Observability

**Commit 14: Add model readiness to `/health` endpoint**
- `/health` response gains a `model_ready` boolean via `ModelManager.is_loaded()`.

### Phase 5 — Tests

**Commit 15: Add tests for new modules**
- `test_content_detector.py`: magazine patterns (Telugu + English), book fallback, empty text, boundary cases.
- `test_confidence_scorer.py`: book confidence calculation, magazine confidence, null/partial fields.
- `test_prompts.py`: verify constants are valid, contain expected JSON field names.
- `test_model_manager.py`: thread-safety under concurrent access (using `ThreadPoolExecutor`), singleton lifecycle, `is_loaded` states.
- `test_llm_backend.py`: interface contract, correct `SamplingParams` per backend.

**Commit 16: Update existing tests for refactored modules**
- `test_vlm_client.py`: update `_reset_vlm()` for ModelManager, add thread-safety tests.
- `test_pipeline.py`: update mock paths to patch new module locations, add sync facade tests, add error propagation tests.
- `test_main.py`: update mock paths, test `/health` model_ready field.
- `test_tasks.py`: test in-flight tracking prevents stale cleanup.
- `test_async_integration.py`: verify queue routing consistency.

## Decision Document

### Modules to be created

| Module | Responsibility |
|--------|---------------|
| `prompts.py` | Prompt template constants |
| `content_detector.py` | Content type detection (book vs magazine) |
| `confidence_scorer.py` | Confidence score calculation |
| `llm_backend.py` | LLM abstraction interface + implementations |
| `model_manager.py` | Thread-safe singleton model lifecycle |
| `exceptions.py` | Typed error classes |

### Modules to be modified

| Module | Changes |
|--------|---------|
| `pipeline.py` | Delegate to new modules, add sync facade, add error boundaries |
| `vlm_client.py` | Use ModelManager internally, use prompts from prompts.py |
| `main.py` | Remove runtime queue routing, update health check, rename `load_llm` |
| `tasks.py` | Add in-flight tracking, use sync facade, rename `load_llm` |
| `celery_config.py` | Unified routing single source of truth |

### Key architectural decisions

1. **Singleton pattern preserved but hardened**: `ModelManager` wraps the vLLM engine with `threading.Lock`. The process-level isolation of Celery prefork workers means each worker has its own singleton. The lock protects against races within a single process (e.g., uvicorn threaded workers). The model stays loaded until the process exits, matching the current behavior.

2. **Prompts stay in code**: Extracted to module-level constants in `prompts.py`. Version-controlled alongside code. No external config files.

3. **Sync Celery kept**: Pipeline gains sync facade methods wrapping `asyncio.run()`. Celery tasks use these. The async originals remain for FastAPI direct use.

4. **Backward-compatible**: All CLI flags, HTTP endpoints, task names, and response schemas remain unchanged.

5. **No Docker changes**: Dockerfile, docker-compose, and deployment config are out of scope.

6. **ModelManager handles heterogeneous workers**: Each Celery worker process has its own `ModelManager` singleton. GPU workers (listening on `vlm_queue`) will load the model once. CPU workers (listening on `default_queue`) will never load it.

## Testing Decisions

### Testing philosophy
- Test external behavior, not implementation details.
- A good test exercises a function/class through its public API with realistic inputs and asserts on the output.
- Thread-safety tests use `concurrent.futures.ThreadPoolExecutor` to hammer shared resources under contention.
- Pure functions (content detection, confidence scoring) are tested with table-driven inputs.
- Pipeline integration tests mock the LLM layer and test the orchestration logic.

### Modules to be tested

| Module | Type of tests | Prior art |
|--------|--------------|-----------|
| `content_detector.py` | Pure function tests with Telugu/English magazine patterns, edge cases | `test_validation.py` (pure function table tests) |
| `confidence_scorer.py` | Pure function tests with partial/full/null data | `test_pipeline.py`'s `calculate_confidence` tests |
| `prompts.py` | Constant validation (expected field names present) | `test_models.py` (schema validation) |
| `model_manager.py` | Thread-safety (concurrent access via `ThreadPoolExecutor`), singleton lifecycle, `is_loaded` | `test_vlm_client.py`'s singleton reset pattern |
| `llm_backend.py` | Interface contract, `SamplingParams` construction per backend | `test_vlm_client.py` mock LLM pattern |
| Updated `pipeline.py` | Sync facade methods, error propagation, delegation to new modules | Existing `test_pipeline.py` patterns |
| Updated `tasks.py` | In-flight tracking prevents cleanup of active files | Existing `test_tasks.py` |
| Updated `health` | `model_ready` field in response | Existing `test_main.py` endpoint tests |

### What won't work as unit tests
- End-to-end tests with real vLLM model loading are excluded (require GPU, take minutes).
- Tests that verify the exact content of AI-generated output (we test that JSON is parseable, not the specific values).

## Out of Scope

- Dockerfile / docker-compose changes
- CI/CD pipeline changes (GitLab CI, pre-commit config)
- Converting test fixtures to shared `conftest.py`
- Adding a coverage threshold gate (`--cov-fail-under`)
- Performance optimization (beyond thread-safety fixes)
- Schema changes to the `ExtractionResult` / API response format
- Adding new extraction capabilities or models
- Kubernetes deployment manifests
- Task queue observability (e.g., Prometheus metrics, structured logging)

## Further Notes

- The `celery_config.py` routing change (Commit 10) should be coordinated with the Docker Compose service definitions to ensure `worker-gpu` and `worker-cpu` still get the right tasks.
- Thread-safety tests (Commit 15) should run in CI but can be skipped manually during development since they're timing-sensitive.
- The sync facade pattern (Commit 12) follows existing Celery conventions and avoids introducing new dependencies.
