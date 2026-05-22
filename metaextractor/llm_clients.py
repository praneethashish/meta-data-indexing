import importlib
import json
import logging
import os
from typing import Any, cast

from . import config
from .config import Settings
from .prompts import BOOK_EXTRACTION_PROMPT, DEFAULT_LANGUAGE_LABEL, LANGUAGE_LABELS, MAGAZINE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


def _settings() -> Settings:
    return config.settings


def build_extraction_prompt(text: str) -> str:
    return BOOK_EXTRACTION_PROMPT.format(text=text[: _settings().BOOK_EXTRACTION_MAX_LENGTH])


def build_magazine_prompt(text: str, lang: str = "te") -> str:
    lang_label = LANGUAGE_LABELS.get(lang, DEFAULT_LANGUAGE_LABEL)
    return MAGAZINE_EXTRACTION_PROMPT.format(
        lang_label=lang_label, text=text[: _settings().MAGAZINE_EXTRACTION_MAX_LENGTH]
    )


def parse_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return cast(dict[str, Any], json.loads(text[start:end]))
    except json.JSONDecodeError:
        return {}


class BaseLLMClient:
    def extract_semantic_fields(self, text: str) -> dict[str, Any]:
        raise NotImplementedError

    def extract_magazine_fields(self, text: str, lang: str = "te") -> dict[str, Any]:
        raise NotImplementedError


def _import_anyllm():
    return importlib.import_module("anyllm")


class LocalVLLMClient(BaseLLMClient):
    def __init__(self, model_id: str | None = None):
        from .model_client import ModelClient

        self._client = ModelClient.get_instance(model_id=model_id)

    def extract_semantic_fields(self, text: str) -> dict[str, Any]:
        prompt = build_extraction_prompt(text)
        result = self._client.generate_and_extract(prompt, max_tokens=256)
        return result if result is not None else {}

    def extract_magazine_fields(self, text: str, lang: str = "te") -> dict[str, Any]:
        prompt = build_magazine_prompt(text, lang=lang)
        result = self._client.generate_and_extract(prompt, temperature=0.1, max_tokens=512)
        return result if result is not None else {}


class AnyLLMClient(BaseLLMClient):
    def __init__(self, model_id: str | None = None):
        s = _settings()
        model = model_id or s.METAEXTRACTOR_LLM_MODEL
        if not model:
            raise RuntimeError("METAEXTRACTOR_LLM_MODEL must be set when env-configured anyllm inference is used.")

        base_url = s.METAEXTRACTOR_LLM_BASE_URL
        api_key = s.METAEXTRACTOR_LLM_API_KEY

        if not base_url:
            raise RuntimeError("METAEXTRACTOR_LLM_BASE_URL must be set when env-configured anyllm inference is used.")
        if not api_key:
            raise RuntimeError("METAEXTRACTOR_LLM_API_KEY must be set when env-configured anyllm inference is used.")

        try:
            anyllm = _import_anyllm()
        except ImportError as exc:
            raise RuntimeError("The env-configured anyllm path requires the 'anyllm' package to be installed.") from exc

        self._model = model
        self._anyllm = anyllm
        self._provider = s.METAEXTRACTOR_LLM_PROVIDER or "openai"
        self._prefixed_model = f"{self._provider}/{self._model}"
        self._configure_anyllm(base_url=base_url, api_key=api_key)

    def _configure_anyllm(self, base_url: str, api_key: str) -> None:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = api_key

        config = getattr(self._anyllm, "get_config", lambda: None)()
        if config is not None and hasattr(config, "set"):
            config.set("openai_base_url", base_url)
            config.set("openai_api_key", api_key)

    def extract_semantic_fields(self, text: str) -> dict[str, Any]:
        prompt = build_extraction_prompt(text)
        return self._call_llm(prompt)

    def extract_magazine_fields(self, text: str, lang: str = "te") -> dict[str, Any]:
        prompt = build_magazine_prompt(text, lang=lang)
        return self._call_llm(prompt, temperature=0.1, max_tokens=512)

    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> dict[str, Any]:
        try:
            response = self._anyllm.chat(
                prompt,
                model=self._prefixed_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["```"],
            )
        except Exception:
            logger.exception("Remote LLM call failed")
            return {}

        content = getattr(response, "content", response)
        if content is None:
            return {}
        return parse_json_object(content)


def has_remote_llm_config() -> bool:
    s = _settings()
    return bool(s.METAEXTRACTOR_LLM_MODEL and s.METAEXTRACTOR_LLM_BASE_URL and s.METAEXTRACTOR_LLM_API_KEY)


def has_partial_remote_llm_config() -> bool:
    s = _settings()
    values = [
        s.METAEXTRACTOR_LLM_MODEL,
        s.METAEXTRACTOR_LLM_BASE_URL,
        s.METAEXTRACTOR_LLM_API_KEY,
    ]
    return any(values) and not all(values)


def create_llm_client(model_id: str | None = None, backend: str = "auto") -> BaseLLMClient:
    if backend == "local":
        return LocalVLLMClient(model_id=model_id)

    if has_partial_remote_llm_config():
        raise RuntimeError(
            "Incomplete remote LLM configuration. Set METAEXTRACTOR_LLM_MODEL, "
            "METAEXTRACTOR_LLM_BASE_URL, and METAEXTRACTOR_LLM_API_KEY together."
        )

    if backend == "remote" or has_remote_llm_config():
        return AnyLLMClient(model_id=model_id)

    return LocalVLLMClient(model_id=model_id)
