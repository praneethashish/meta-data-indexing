import json
import logging
import os
from typing import Any, cast

from PIL import Image

try:
    from vllm import SamplingParams
except ImportError:

    class _SamplingParamsStub:
        def __init__(self, **kwargs: Any) -> None:
            pass

    SamplingParams = _SamplingParamsStub  # type: ignore

from .model_manager import ModelManager
from .prompts import IMAGE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class VLMClient:
    """Unified wrapper for Gemma/Qwen-VL vLLM inference.

    Delegates model lifecycle management (loading, caching, thread-safe
    access) to ModelManager. Use get_instance() for the shared singleton.
    """

    _instance: "VLMClient | None" = None

    def __init__(self, model_id: str | None = None, max_model_len: int = 4096):
        """Initialize the VLMClient. Note: Use get_instance() for shared singleton."""
        self.model_id = (
            model_id or os.getenv("VLLM_MODEL_ID") or os.getenv("VLLM_MODEL") or self._detect_cached_model()
        )
        self.max_model_len = max_model_len
        self._model_manager = ModelManager.get_instance()
        self._model_manager.get_or_create(
            model_id=self.model_id,
            max_model_len=self.max_model_len,
        )

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

    def generate(self, prompts: list[str], sampling_params: Any = None) -> list[Any]:
        """Generate text from one or more prompts."""
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

        model = self._model_manager.get_model()
        if model is None:
            raise RuntimeError("vLLM engine not initialized")

        return cast(list[Any], model.generate(prompts, sampling_params))

    async def describe_image(self, image_path: str) -> dict[str, Any]:
        """Generate a rich description for an image using vLLM multimodal inference."""
        image = Image.open(image_path).convert("RGB")

        prompt_text = IMAGE_ANALYSIS_PROMPT

        model = self._model_manager.get_model()
        if model is None:
            raise RuntimeError("vLLM engine not initialized")

        sampling_params = SamplingParams(temperature=0.2, max_tokens=512, stop=["```"])
        inputs = {
            "prompt": f"<|image_1|>\n{prompt_text}",
            "multi_modal_data": {"image": image},
        }

        outputs = model.generate([inputs], sampling_params)
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