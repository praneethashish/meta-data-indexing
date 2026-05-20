import json
import logging
from typing import Any, cast

from PIL import Image

try:
    from vllm import SamplingParams
except ImportError:

    class _SamplingParamsStub:
        def __init__(self, **kwargs: Any) -> None:
            pass

    SamplingParams = _SamplingParamsStub  # type: ignore

from .config import settings
from .model_manager import ModelManager
from .prompts import IMAGE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


def parse_json_from_text(text_out: str) -> dict[str, Any] | None:
    """Extract and parse the first JSON object found in text."""
    start = text_out.find("{")
    end = text_out.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return cast(dict[str, Any], json.loads(text_out[start:end]))
    except json.JSONDecodeError:
        return None


class ModelClient:
    """Unified client for the vLLM model engine.

    Handles text generation, multimodal image description, JSON
    extraction, and model lifecycle management. Singleton per process
    via get_instance().
    """

    _instance: "ModelClient | None" = None

    def __init__(self, model_id: str | None = None, max_model_len: int = 4096):
        self.model_id = model_id or settings.VLLM_MODEL_ID or settings.VLLM_MODEL
        self.max_model_len = max_model_len
        self._model_manager = ModelManager.get_instance()
        self._model_manager.get_or_create(
            model_id=self.model_id,
            max_model_len=self.max_model_len,
        )
        if self.model_id is None:
            self.model_id = self._model_manager.model_id

    @classmethod
    def get_instance(cls, model_id: str | None = None, max_model_len: int = 4096) -> "ModelClient":
        if cls._instance is None:
            cls._instance = cls(model_id=model_id, max_model_len=max_model_len)
        return cls._instance

    def is_ready(self) -> bool:
        return self._model_manager.is_loaded()

    def generate(self, prompts: list[str], sampling_params: Any = None) -> list[Any]:
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=settings.DEFAULT_TEMPERATURE,
                max_tokens=settings.DEFAULT_MAX_TOKENS,
                stop=settings.DEFAULT_STOP,
            )
        model = self._model_manager.get_model()
        if model is None:
            raise RuntimeError("vLLM engine not initialized")
        return cast(list[Any], model.generate(prompts, sampling_params))

    def generate_and_extract(
        self, prompt: str, temperature: float | None = None, max_tokens: int | None = None
    ) -> dict[str, Any] | None:
        params_kwargs: dict[str, Any] = {
            "temperature": temperature if temperature is not None else settings.DEFAULT_TEMPERATURE,
            "max_tokens": max_tokens if max_tokens is not None else settings.DEFAULT_MAX_TOKENS,
            "stop": settings.DEFAULT_STOP,
        }
        sampling_params = SamplingParams(**params_kwargs)
        try:
            outputs = self.generate([prompt], sampling_params)
            text_out = outputs[0].outputs[0].text.strip()
            return parse_json_from_text(text_out)
        except Exception as e:
            logger.warning(f"generate_and_extract failed: {e}")
            return None

    async def describe_image(self, image_path: str) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")

        prompt_text = IMAGE_ANALYSIS_PROMPT

        model = self._model_manager.get_model()
        if model is None:
            raise RuntimeError("vLLM engine not initialized")

        sampling_params = SamplingParams(
            temperature=settings.VISION_TEMPERATURE, max_tokens=settings.VISION_MAX_TOKENS, stop=settings.DEFAULT_STOP
        )
        inputs = {
            "prompt": f"<|image_1|>\n{prompt_text}",
            "multi_modal_data": {"image": image},
        }

        outputs = model.generate([inputs], sampling_params)
        text_out = outputs[0].outputs[0].text.strip()

        result = parse_json_from_text(text_out)
        if result is not None:
            return result

        logger.warning("Failed to parse JSON from describe_image output.")
        return {
            "description": text_out[: settings.DESCRIPTION_FALLBACK_LENGTH] if text_out else None,
            "text_content": None,
            "language": None,
            "scene_classification": "other",
            "entities": [],
        }

    @classmethod
    def reset(cls) -> None:
        """Reset singleton state. For testing only."""
        cls._instance = None
        ModelManager.reset()
