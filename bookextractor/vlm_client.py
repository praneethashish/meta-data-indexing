import logging
import os
from typing import Any, cast

try:
    from vllm import LLM, SamplingParams
except ImportError:
    # Fallback for linting/testing in slim environments
    class _SamplingParamsStub:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class _LLMStub:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def generate(self, *_args: Any, **_kwargs: Any) -> Any:
            return []

    SamplingParams = _SamplingParamsStub  # type: ignore
    LLM = _LLMStub  # type: ignore

from .hardware import get_vllm_config

logger = logging.getLogger(__name__)


class VLMClient:
    """
    Unified wrapper for Gemma/Qwen-VL vLLM inference.
    Handles hardware-optimized initialization and provides methods for text and vision tasks.
    """

    _instance: "VLMClient | None" = None
    _llm: Any = None

    def __init__(self, model_id: str | None = None, max_model_len: int = 4096):
        """
        Initialize the VLMClient. Note: Use get_instance() for shared LLM resource.
        """
        if VLMClient._llm is None:
            self.model_id = (
                model_id or os.getenv("VLLM_MODEL_ID") or os.getenv("VLLM_MODEL") or self._detect_cached_model()
            )
            self.max_model_len = max_model_len
            self._initialize_llm()

    @staticmethod
    def _detect_cached_model() -> str:
        from .models_registry import AVAILABLE_MODELS, get_cached_models

        cached = get_cached_models()
        if not cached:
            return "Qwen/Qwen2.5-VL-7B-Instruct"

        registry_ids = {m["id"] for m in AVAILABLE_MODELS}
        known_cached = [m for m in cached if m in registry_ids]

        if len(known_cached) == 1:
            logger.info(f"Auto-detected cached model: {known_cached[0]}")
            return known_cached[0]

        if known_cached:
            logger.info(f"Multiple cached models. Using: {known_cached[0]}")
            return known_cached[0]

        logger.info(f"No known cached models. Using: {cached[0]}")
        return cached[0]

    @classmethod
    def get_instance(cls, model_id: str | None = None, max_model_len: int = 4096) -> "VLMClient":
        """Get or create a singleton instance of VLMClient."""
        if cls._instance is None:
            cls._instance = cls(model_id=model_id, max_model_len=max_model_len)
        return cls._instance

    def _initialize_llm(self) -> None:
        """Initialize vLLM engine with hardware-optimized configuration."""
        config = get_vllm_config()

        logger.info(f"Initializing VLMClient with hardware: {config['detected_hardware']['name']}")
        logger.info(
            f"Applying vLLM config: dtype={config['dtype']}, "
            f"tp_size={config['tensor_parallel_size']}, "
            f"memory_util={config['gpu_memory_utilization']}"
        )

        # Set environment variable to force device type (fixes vLLM auto-detection issues)
        os.environ["VLLM_TARGET_DEVICE"] = config["device"]

        VLMClient._llm = LLM(
            model=self.model_id,
            tensor_parallel_size=config["tensor_parallel_size"],
            dtype=config["dtype"],
            max_model_len=self.max_model_len,
            gpu_memory_utilization=config["gpu_memory_utilization"],
            trust_remote_code=True,
        )

    def generate(self, prompts: list[str], sampling_params: Any = None) -> list[Any]:
        """Generate text from one or more prompts."""
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

        if VLMClient._llm is None:
            raise RuntimeError("vLLM engine not initialized")

        return cast(list[Any], VLMClient._llm.generate(prompts, sampling_params))

    async def describe_image(self, image_path: str) -> dict[str, Any]:
        """
        Generate description for an image.
        Note: Currently a placeholder until full Phase 2 vision processing is implemented.
        """
        # In a future update, this will handle image encoding and multi-modal prompt generation
        # for models like Qwen2-VL or Gemma-VL.
        _ = image_path  # Silence unused argument warning
        return {
            "description": "Image description not yet implemented in VLMClient wrapper.",
            "scene_classification": "other",
            "text_content": None,
        }
