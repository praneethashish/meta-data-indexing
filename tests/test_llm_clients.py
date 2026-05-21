from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bookextractor.llm_clients import (
    AnyLLMClient,
    create_llm_client,
    has_partial_remote_llm_config,
    has_remote_llm_config,
    parse_json_object,
)


def test_parse_json_object_extracts_embedded_json():
    result = parse_json_object('prefix {"title": "Test Book"} suffix')

    assert result == {"title": "Test Book"}


def test_parse_json_object_returns_empty_dict_on_invalid_json():
    result = parse_json_object("not json at all")

    assert result == {}


def test_has_remote_llm_config_returns_false_when_unset(monkeypatch):
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_MODEL", raising=False)
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_API_KEY", raising=False)

    from bookextractor import config

    config.settings = config.Settings()

    assert has_remote_llm_config() is False


def test_has_remote_llm_config_requires_complete_values(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "test-model")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_API_KEY", raising=False)

    from bookextractor import config

    config.settings = config.Settings()

    assert has_remote_llm_config() is False


def test_has_partial_remote_llm_config_detects_incomplete_values(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "test-model")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_API_KEY", raising=False)

    from bookextractor import config

    config.settings = config.Settings()

    assert has_partial_remote_llm_config() is True


def test_create_llm_client_uses_anyllm_when_remote_config_exists(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "remote-model")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_API_KEY", "test-key")

    from bookextractor import config

    config.settings = config.Settings()

    with patch("bookextractor.llm_clients.AnyLLMClient") as mock_client:
        client = create_llm_client(model_id="remote-model")

    assert client == mock_client.return_value
    mock_client.assert_called_once_with(model_id="remote-model")


def test_create_llm_client_falls_back_to_local_vllm(monkeypatch):
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_MODEL", raising=False)
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_API_KEY", raising=False)

    from bookextractor import config

    config.settings = config.Settings()

    with patch("bookextractor.llm_clients.LocalVLLMClient") as mock_client:
        client = create_llm_client()

    assert client == mock_client.return_value
    mock_client.assert_called_once_with(model_id=None)


def test_create_llm_client_rejects_incomplete_remote_config(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "remote-model")
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_API_KEY", "test-key")

    from bookextractor import config

    config.settings = config.Settings()

    with pytest.raises(RuntimeError, match="Incomplete remote LLM configuration"):
        create_llm_client()


def test_anyllm_requires_model(monkeypatch):
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_MODEL", raising=False)
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_API_KEY", "test-key")

    from bookextractor import config

    config.settings = config.Settings()

    with pytest.raises(RuntimeError, match="BOOKEXTRACTOR_LLM_MODEL"):
        AnyLLMClient()


def test_anyllm_requires_base_url(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "test-model")
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_API_KEY", "test-key")

    from bookextractor import config

    config.settings = config.Settings()

    with pytest.raises(RuntimeError, match="BOOKEXTRACTOR_LLM_BASE_URL"):
        AnyLLMClient()


def test_anyllm_requires_api_key(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "test-model")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.delenv("BOOKEXTRACTOR_LLM_API_KEY", raising=False)

    from bookextractor import config

    config.settings = config.Settings()

    with pytest.raises(RuntimeError, match="BOOKEXTRACTOR_LLM_API_KEY"):
        AnyLLMClient()


def test_anyllm_extracts_semantic_fields(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "test-model")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_API_KEY", "test-key")

    from bookextractor import config

    config.settings = config.Settings()

    mock_chat = MagicMock(
        return_value=SimpleNamespace(content='{"title": "Remote Book", "author": "Remote Author", "publisher": null}')
    )
    mock_config = MagicMock()

    mock_anyllm = SimpleNamespace(chat=mock_chat, get_config=MagicMock(return_value=mock_config))

    with patch("bookextractor.llm_clients._import_anyllm", return_value=mock_anyllm):
        client = AnyLLMClient()
        result = client.extract_semantic_fields("OCR content")

    assert result == {"title": "Remote Book", "author": "Remote Author", "publisher": None}
    mock_chat.assert_called_once()
    mock_config.set.assert_any_call("openai_base_url", "http://localhost:1234/v1")
    mock_config.set.assert_any_call("openai_api_key", "test-key")


def test_anyllm_requires_package(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "test-model")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_API_KEY", "test-key")

    from bookextractor import config

    config.settings = config.Settings()

    with (
        patch("bookextractor.llm_clients._import_anyllm", side_effect=ImportError("missing")),
        pytest.raises(RuntimeError, match="requires the 'anyllm' package"),
    ):
        AnyLLMClient()


def test_anyllm_returns_empty_dict_when_chat_fails(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "test-model")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_API_KEY", "test-key")

    from bookextractor import config

    config.settings = config.Settings()

    mock_anyllm = SimpleNamespace(
        chat=MagicMock(side_effect=RuntimeError("boom")),
        get_config=MagicMock(return_value=MagicMock()),
    )

    with patch("bookextractor.llm_clients._import_anyllm", return_value=mock_anyllm):
        client = AnyLLMClient()

    assert client.extract_semantic_fields("OCR content") == {}


def test_anyllm_returns_empty_dict_when_response_has_no_content(monkeypatch):
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_MODEL", "test-model")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("BOOKEXTRACTOR_LLM_API_KEY", "test-key")

    from bookextractor import config

    config.settings = config.Settings()

    mock_anyllm = SimpleNamespace(
        chat=MagicMock(return_value=SimpleNamespace(content=None)),
        get_config=MagicMock(return_value=MagicMock()),
    )

    with patch("bookextractor.llm_clients._import_anyllm", return_value=mock_anyllm):
        client = AnyLLMClient()

    assert client.extract_semantic_fields("OCR content") == {}
