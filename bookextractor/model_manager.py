import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe singleton manager for the vLLM engine lifecycle.

    Provides locking around initialization and access so that concurrent
    callers (e.g. multiple uvicorn threads or async event loops) do not
    race to load the same model or access an incompletely-initialized engine.

    Usage:
        manager = ModelManager()
        config = get_vllm_config()
        model = manager.get_or_create(model_id="Qwen/Qwen2.5-VL-7B-Instruct", ...)
        if manager.is_loaded():
            ...
    """

    _instance: "ModelManager | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: str | None = None
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Get the singleton ModelManager instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. For testing only."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._model = None
                cls._instance._model_id = None
                cls._instance = None

    def get_or_create(
        self,
        model_id: str | None = None,
        max_model_len: int = 4096,
        vllm_config: dict[str, Any] | None = None,
    ) -> Any:
        """Get the existing model or initialize a new one.

        Thread-safe: only one caller will initialize; others will wait
        and receive the already-loaded model.
        """
        with self._init_lock:
            if self._model is not None:
                return self._model

            from .hardware import get_vllm_config

            if vllm_config is None:
                vllm_config = get_vllm_config()

            if model_id is None:
                from .models_registry import AVAILABLE_MODELS, get_cached_models

                cached = get_cached_models()
                if cached:
                    registry_ids = {m["id"] for m in AVAILABLE_MODELS}
                    known_cached = [m for m in cached if m in registry_ids]
                    model_id = known_cached[0] if known_cached else cached[0]
                else:
                    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
                    logger.info(f"No cached model found. Using default: {model_id}")

            import os

            try:
                from vllm import LLM
            except ImportError:
                raise RuntimeError(
                    "vLLM is not installed. For LLM-based extraction (PDF, text files), "
                    "install the ML dependencies:\n"
                    "  uv pip install -e '.[ml]'\n"
                    "or\n"
                    "  uv sync --extra ml"
                ) from None

            logger.info(f"Initializing model: {model_id}")
            logger.info(
                f"Applying vLLM config: dtype={vllm_config['dtype']}, "
                f"tp_size={vllm_config['tensor_parallel_size']}, "
                f"memory_util={vllm_config['gpu_memory_utilization']}"
            )

            if "VLLM_TARGET_DEVICE" not in os.environ:
                os.environ["VLLM_TARGET_DEVICE"] = vllm_config["device"]

            self._model = LLM(
                model=model_id,
                tensor_parallel_size=vllm_config["tensor_parallel_size"],
                dtype=vllm_config["dtype"],
                max_model_len=max_model_len,
                gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
                trust_remote_code=True,
            )
            self._model_id = model_id
            logger.info(f"Model {model_id} loaded successfully.")
            return self._model

    def get_model(self) -> Any | None:
        """Get the model if loaded, without initializing."""
        return self._model

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        """Return the model ID of the loaded model, or None."""
        return self._model_id