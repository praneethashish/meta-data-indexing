import json
import logging
import os
from typing import Any, cast

from PIL import Image

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
        if not isinstance(LLM, type) or LLM.__name__ == "_LLMStub":
            raise RuntimeError(
                "vLLM is not installed. For LLM-based extraction (PDF, text files), "
                "install the ML dependencies:\n"
                "  uv pip install -e '.[ml]'\n"
                "or\n"
                "  uv sync --extra ml"
            )

        config = get_vllm_config()

        logger.info(f"Initializing VLMClient with hardware: {config['detected_hardware']['name']}")
        logger.info(
            f"Applying vLLM config: dtype={config['dtype']}, "
            f"tp_size={config['tensor_parallel_size']}, "
            f"memory_util={config['gpu_memory_utilization']}"
        )

        # Set environment variable to force device type only if not already configured
        if "VLLM_TARGET_DEVICE" not in os.environ:
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
        """Generate a rich description for an image using vLLM multimodal inference."""
        image = Image.open(image_path).convert("RGB")

        prompt_text = (
            "Analyze this image. Return ONLY a valid JSON object with the following keys: "
            "'description' (string, visual description), 'text_content' (string, any visible text, or null), "
            "'language' (string, detected language, or null), "
            "'scene_classification' (string, e.g., 'document', 'nature', 'diagram'), "
            "'entities' (list of dicts with 'name' and 'type')."
        )

        if VLMClient._llm is None:
            raise RuntimeError("vLLM engine not initialized")

        sampling_params = SamplingParams(temperature=0.2, max_tokens=512, stop=["```"])
        inputs = {
            "prompt": f"<|image_1|>\n{prompt_text}",
            "multi_modal_data": {"image": image},
        }

        outputs = VLMClient._llm.generate([inputs], sampling_params)
        text_out = outputs[0].outputs[0].text.strip()

        start = text_out.find("{")
        end = text_out.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return cast(dict[str, Any], json.loads(text_out[start:end]))
            except json.JSONDecodeError as e:
                logger.error(f"VLM describe_image yielded invalid JSON: {e}. Raw text: {text_out}")
        else:
            logger.warning("Failed to locate JSON brackets in VLM describe_image output.")

        return {
            "description": text_out[:200] if text_out else None,
            "text_content": None,
            "language": None,
            "scene_classification": "other",
            "entities": [],
        }
