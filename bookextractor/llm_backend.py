import json
import logging
from abc import ABC, abstractmethod
from typing import Any, cast

try:
    from vllm import SamplingParams
except ImportError:

    class _SamplingParamsStub:
        def __init__(self, **kwargs: Any) -> None:
            pass

    SamplingParams = _SamplingParamsStub  # type: ignore

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract interface for LLM inference backends."""

    @abstractmethod
    def generate(self, prompts: list[str], sampling_params: Any = None) -> list[Any]:
        """Generate text from one or more text prompts."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the backend is loaded and ready for inference."""
        ...


class TextLLMBackend(LLMBackend):
    """LLM backend for text-only metadata extraction."""

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_STOP = ["```"]

    def __init__(self, model: Any):
        self._model = model

    def generate(self, prompts: list[str], sampling_params: Any = None) -> list[Any]:
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.DEFAULT_TEMPERATURE,
                max_tokens=self.DEFAULT_MAX_TOKENS,
                stop=self.DEFAULT_STOP,
            )
        if self._model is None:
            raise RuntimeError("TextLLMBackend: model is not loaded")
        return cast(list[Any], self._model.generate(prompts, sampling_params))

    def is_ready(self) -> bool:
        return self._model is not None

    def extract_json_from_output(self, text_out: str) -> dict[str, Any] | None:
        """Extract the first JSON object from LLM output text."""
        start = text_out.find("{")
        end = text_out.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return cast(dict[str, Any], json.loads(text_out[start:end]))
            except json.JSONDecodeError as e:
                logger.error(f"LLM output yielded invalid JSON: {e}. Raw text: {text_out[:200]}")
                return None
        logger.warning("Failed to locate JSON brackets in LLM output.")
        return None


class VisionLLMBackend(LLMBackend):
    """LLM backend for multimodal (vision + text) inference."""

    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_STOP = ["```"]

    def __init__(self, model: Any):
        self._model = model

    def generate(self, prompts: list[str], sampling_params: Any = None) -> list[Any]:
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.DEFAULT_TEMPERATURE,
                max_tokens=self.DEFAULT_MAX_TOKENS,
                stop=self.DEFAULT_STOP,
            )
        if self._model is None:
            raise RuntimeError("VisionLLMBackend: model is not loaded")
        return cast(list[Any], self._model.generate(prompts, sampling_params))

    def is_ready(self) -> bool:
        return self._model is not None