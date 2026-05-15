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
        result = parse_json_from_text(text_out)
        if result is not None:
            return result
        logger.warning("Failed to locate or parse JSON brackets in LLM output.")
        return None

    def generate_and_extract(
        self, prompt: str, temperature: float | None = None, max_tokens: int | None = None
    ) -> dict[str, Any] | None:
        """Generate text from a prompt and extract JSON from the output.

        Returns the parsed JSON dict, or None if generation/extraction failed.
        """
        params_kwargs: dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
            "max_tokens": max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
            "stop": self.DEFAULT_STOP,
        }
        sampling_params = SamplingParams(**params_kwargs)
        try:
            outputs = self.generate([prompt], sampling_params)
            text_out = outputs[0].outputs[0].text.strip()
            return self.extract_json_from_output(text_out)
        except Exception as e:
            logger.warning(f"generate_and_extract failed: {e}")
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