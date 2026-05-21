import logging
import os
import sys
from pathlib import Path
from typing import ClassVar


def _optional_float(env_var: str) -> float | None:
    val = os.getenv(env_var)
    return float(val) if val is not None else None


def _optional_int(env_var: str) -> int | None:
    val = os.getenv(env_var)
    return int(val) if val is not None else None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


class Settings:
    """Centralized application configuration.

    Reads from environment variables with sensible defaults.
    Access via module-level ``settings`` singleton.
    """

    # ── File type rules (centralized, not env-overridable) ─
    ALLOWED_EXTENSIONS: ClassVar[tuple[str, ...]] = (
        ".pdf",
        ".md",
        ".json",
        ".txt",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tiff",
        ".tif",
    )
    IMAGE_EXTENSIONS: ClassVar[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif")

    # ── Book confidence scoring thresholds ─────────────────
    BOOK_CONFIDENCE_FIELDS: ClassVar[list[str]] = ["title", "author", "publisher", "published_date"]
    MAGAZINE_CONFIDENCE_THRESHOLDS: ClassVar[dict[str, float]] = {
        "magazine_name": 0.8,
        "editor": 0.7,
        "publisher": 0.7,
        "issue_date": 0.8,
        "issue_number": 0.6,
        "price": 0.6,
    }

    # ── Magazine detection patterns ────────────────────────
    MAGAZINE_PATTERNS: ClassVar[list[str]] = [
        "\u0c2e\u0c3e\u0c38\u0c2a\u0c24\u0c4d\u0c30\u0c3f\u0c15",
        "\u0c38\u0c02\u0c1a\u0c3f\u0c15",
        "\u0c1a\u0c02\u0c26\u0c3e",
        "\u0c0f\u0c1c\u0c02\u0c1f\u0c4d\u0c32\u0c41",
        "magazine",
        "issue",
        "vol.",
        "no.",
        "subscription",
        "monthly",
        "periodical",
    ]

    def __init__(self) -> None:
        # ── Paths ──────────────────────────────────────────────
        self.UPLOAD_DIR: str = os.getenv("BOOKEXTRACTOR_UPLOAD_DIR", "uploads")
        self.MODELS_DIR: str = os.getenv(
            "BOOKEXTRACTOR_MODELS_DIR", str(Path(__file__).resolve().parents[1] / "models")
        )
        self.HF_HOME: str = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

        # ── API ────────────────────────────────────────────────
        self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT: int = int(os.getenv("API_PORT", "8000"))

        # ── vLLM Model ─────────────────────────────────────────
        self.VLLM_MODEL_ID: str | None = os.getenv("VLLM_MODEL_ID")
        self.VLLM_MODEL: str | None = os.getenv("VLLM_MODEL")
        self.VLLM_DEVICE: str | None = os.getenv("VLLM_DEVICE")
        self.VLLM_DTYPE: str | None = os.getenv("VLLM_DTYPE")
        self.VLLM_GPU_MEMORY_UTILIZATION: float | None = _optional_float("VLLM_GPU_MEMORY_UTILIZATION")
        self.VLLM_TARGET_DEVICE: str | None = os.getenv("VLLM_TARGET_DEVICE")
        self.VLLM_TENSOR_PARALLEL_SIZE: int | None = _optional_int("VLLM_TENSOR_PARALLEL_SIZE")
        self.MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "4096"))

        # ── Remote LLM (anyLLM) ────────────────────────────────
        self.BOOKEXTRACTOR_LLM_MODEL: str | None = os.getenv("BOOKEXTRACTOR_LLM_MODEL")
        self.BOOKEXTRACTOR_LLM_BASE_URL: str | None = os.getenv("BOOKEXTRACTOR_LLM_BASE_URL")
        self.BOOKEXTRACTOR_LLM_API_KEY: str | None = os.getenv("BOOKEXTRACTOR_LLM_API_KEY")
        self.BOOKEXTRACTOR_LLM_PROVIDER: str | None = os.getenv("BOOKEXTRACTOR_LLM_PROVIDER")

        # ── Text Extraction Defaults ───────────────────────────
        self.DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        self.DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
        self.DEFAULT_STOP: list[str] = ["```"]
        self.VISION_TEMPERATURE: float = float(os.getenv("VISION_TEMPERATURE", "0.2"))
        self.VISION_MAX_TOKENS: int = int(os.getenv("VISION_MAX_TOKENS", "512"))

        # ── Text Snippet Lengths ───────────────────────────────
        self.TEXT_SNIPPET_LENGTH: int = 500
        self.BOOK_EXTRACTION_MAX_LENGTH: int = 15000
        self.MAGAZINE_EXTRACTION_MAX_LENGTH: int = 3000
        self.DESCRIPTION_FALLBACK_LENGTH: int = 200

        # ── Celery ─────────────────────────────────────────────
        self.CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        self.CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
        self.CELERY_TASK_TIME_LIMIT: int = int(os.getenv("CELERY_TASK_TIME_LIMIT", "3600"))
        self.CELERY_TASK_SOFT_TIME_LIMIT: int = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "3000"))
        self.CELERY_WORKER_PREFETCH_MULTIPLIER: int = int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1"))
        self.CELERY_WORKER_MAX_TASKS_PER_CHILD: int = int(os.getenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "100"))
        self.CELERY_RESULT_EXPIRES: int = int(os.getenv("CELERY_RESULT_EXPIRES", "86400"))
        self.CELERY_DEFAULT_QUEUE: str = os.getenv("CELERY_DEFAULT_QUEUE", "default_queue")
        self.CELERY_TASK_ALWAYS_EAGER: bool = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true"

        # ── Worker ─────────────────────────────────────────────
        self.STALE_FILE_THRESHOLD: int = int(os.getenv("STALE_FILE_THRESHOLD", "86400"))
        self.DEFAULT_WORKER_CONCURRENCY: int = int(os.getenv("DEFAULT_WORKER_CONCURRENCY", "4"))

        # ── External Services ──────────────────────────────────
        self.VPARSE_API_URL: str = os.getenv("VPARSE_API_URL", "http://localhost:8000/file_parse")
        self.VPARSE_TIMEOUT: float = float(os.getenv("VPARSE_TIMEOUT", "900"))
        self.OPENLIBRARY_TIMEOUT: float = float(os.getenv("OPENLIBRARY_TIMEOUT", "10"))


settings = Settings()
