from unittest.mock import MagicMock, patch

import pytest

from bookextractor.model_client import ModelClient, parse_json_from_text
from bookextractor.model_manager import ModelManager

_MOCK_CONFIG = {
    "detected_hardware": {"name": "TestDevice", "memory_gb": 16, "count": 1},
    "device": "cuda",
    "dtype": "float16",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
}


class MockLLM:
    __name__ = "LLM"

    def __init__(self, **_kwargs):
        pass

    generate = MagicMock(return_value=[])


@pytest.fixture
def mock_llm():
    with (
        patch("bookextractor.model_manager.LLM", MockLLM),
        patch("bookextractor.hardware.get_vllm_config", return_value=_MOCK_CONFIG),
    ):
        yield MockLLM


def _reset():
    ModelClient.reset()


def _make_client():
    """Create a ModelClient bypassing __init__ to avoid vLLM loading."""
    client = ModelClient.__new__(ModelClient)
    client.model_id = "test-model"
    client.max_model_len = 4096
    client._model_manager = ModelManager()
    return client


def test_model_client_singleton(mock_llm):  # noqa: ARG001
    _reset()
    client1 = ModelClient.get_instance()
    client2 = ModelClient.get_instance()
    assert client1 is client2
    assert client1.is_ready()
    _reset()


@pytest.mark.asyncio
async def test_model_client_generate(mock_llm):  # noqa: ARG001
    _reset()
    client = ModelClient.get_instance()

    mock_instance = MagicMock()
    mock_output = MagicMock()
    mock_output.text = "generated text"
    mock_instance.generate.return_value = [MagicMock(outputs=[mock_output])]
    client._model_manager._model = mock_instance

    result = client.generate(["prompt"])
    assert result is not None
    mock_instance.generate.assert_called_once()
    _reset()


def test_model_client_is_ready_with_model():
    _reset()
    client = _make_client()
    client._model_manager._model = MagicMock()
    assert client.is_ready() is True


def test_model_client_not_ready_without_model():
    _reset()
    client = _make_client()
    assert client.is_ready() is False


def test_model_client_generate_raises_without_model():
    _reset()
    client = _make_client()
    with pytest.raises(RuntimeError, match="vLLM engine not initialized"):
        client.generate(["prompt"])


def test_parse_json_from_output():
    result = parse_json_from_text('Some text {"key": "value"} more text')
    assert result == {"key": "value"}


def test_parse_json_from_output_empty():
    result = parse_json_from_text("no json here")
    assert result is None


def test_parse_json_from_output_invalid():
    result = parse_json_from_text('{"invalid": json}')
    assert result is None


def test_generate_and_extract():
    _reset()
    client = _make_client()

    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text='{"title": "Test Book", "author": "Test Author"}')]

    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = [mock_output]

    client._model_manager._model = mock_llm_instance

    result = client.generate_and_extract("Extract metadata from this text")
    assert result is not None
    assert result["title"] == "Test Book"
    assert result["author"] == "Test Author"


def test_generate_and_extract_failure():
    _reset()
    client = _make_client()

    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = []

    client._model_manager._model = mock_llm_instance

    result = client.generate_and_extract("Extract metadata")
    assert result is None


@pytest.mark.asyncio
async def test_describe_image(tmp_path):
    _reset()
    client = _make_client()

    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")

    mock_llm_instance = MagicMock()
    mock_output = MagicMock()
    mock_output.outputs = [
        MagicMock(
            text='{"description": "A test image", "text_content": "Hello", "language": "en", "scene_classification": "document", "entities": []}'
        )
    ]
    mock_llm_instance.generate.return_value = [mock_output]

    client._model_manager._model = mock_llm_instance

    with patch("bookextractor.model_client.Image") as mock_pil_image:
        mock_pil_image.open.return_value.convert.return_value = MagicMock()
        result = await client.describe_image(str(image_path))

    assert "description" in result
    assert result["description"] == "A test image"
    assert result["scene_classification"] == "document"


def test_model_client_delegates_model_detection_to_manager():
    _reset()

    class FakeLLM:
        __name__ = "LLM"

        def __init__(self, **_kwargs):
            pass

    with (
        patch("bookextractor.hardware.get_vllm_config", return_value=_MOCK_CONFIG),
        patch("bookextractor.model_manager.LLM", FakeLLM),
    ):
        manager = ModelManager.get_instance()
        manager.get_or_create(model_id=None, max_model_len=4096)
        assert manager.model_id is not None
        assert manager.is_loaded()
    _reset()
