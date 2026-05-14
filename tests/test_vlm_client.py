import os
from unittest.mock import MagicMock, patch

import pytest

try:
    from vllm import SamplingParams as _SamplingParams
except ImportError:
    _SamplingParams = None  # type: ignore

from bookextractor.vlm_client import VLMClient

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
        self.generate = MagicMock(return_value=[])


@pytest.fixture
def mock_llm():
    with (
        patch("bookextractor.vlm_client.LLM", MockLLM),
        patch("bookextractor.vlm_client.get_vllm_config", return_value=_MOCK_CONFIG),
    ):
        yield MockLLM


@pytest.mark.skipif(_SamplingParams is None, reason="vllm not installed")
def test_vlm_client_direct_instantiation(mock_llm):  # noqa: ARG001
    client1 = VLMClient()
    client1.startup()
    client2 = VLMClient()
    client2.startup()

    assert client1 is not client2
    assert client1._llm is not None
    assert client2._llm is not None


@pytest.mark.skipif(_SamplingParams is None, reason="vllm not installed")
@pytest.mark.asyncio
async def test_vlm_client_generate(mock_llm):  # noqa: ARG001
    client = VLMClient()
    client.startup()

    mock_instance = MagicMock()
    mock_output = MagicMock()
    mock_output.text = "generated text"
    mock_instance.generate.return_value = [MagicMock(outputs=[mock_output])]
    client._llm = mock_instance

    result = client.generate(["prompt"])
    assert result is not None
    mock_instance.generate.assert_called_once()

    custom_params = MagicMock()
    client.generate(["prompt"], sampling_params=custom_params)
    assert mock_instance.generate.call_count == 2


@pytest.mark.asyncio
async def test_vlm_client_describe_image(tmp_path):
    """Test describe_image VLM multimodal inference."""
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

    client = VLMClient.__new__(VLMClient)
    client._llm = mock_llm_instance
    client.model_id = "test-model"
    client.max_model_len = 4096

    with patch("bookextractor.vlm_client.Image") as mock_pil_image:
        mock_img = MagicMock()
        mock_img.convert.return_value = MagicMock()
        mock_pil_image.open.return_value.__enter__ = MagicMock(return_value=mock_img)
        mock_pil_image.open.return_value.__exit__ = MagicMock(return_value=False)
        result = await client.describe_image(str(image_path))

    assert "description" in result
    assert result["description"] == "A test image"
    assert result["scene_classification"] == "document"


def test_vlm_client_generate_without_init():
    client = VLMClient.__new__(VLMClient)
    client._llm = None

    with pytest.raises(RuntimeError, match="vLLM engine not initialized"):
        client.generate(["prompt"])


def test_vlm_client_auto_detect_cached():
    with patch("bookextractor.models_registry.get_cached_models", return_value=[]):
        result = VLMClient._detect_cached_model()
        assert result == "Qwen/Qwen2.5-VL-7B-Instruct"

    with patch("bookextractor.models_registry.get_cached_models", return_value=["myorg/mymodel"]):
        result = VLMClient._detect_cached_model()
        assert result == "myorg/mymodel"


def test_vllm_target_device_env_guard(monkeypatch):
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "custom_tpu_device")

    class SingeLLM:
        __name__ = "LLM"

        def __init__(self, **_kwargs):
            pass

    with (
        patch("bookextractor.vlm_client.get_vllm_config", return_value=_MOCK_CONFIG),
        patch("bookextractor.vlm_client.LLM", SingeLLM),
    ):
        client = VLMClient.__new__(VLMClient)
        client.model_id = "test-model"
        client.max_model_len = 4096
        client._initialize_llm()

        assert os.environ["VLLM_TARGET_DEVICE"] == "custom_tpu_device"


def test_vlm_client_startup_shutdown(mock_llm):  # noqa: ARG001
    client = VLMClient()
    client.startup()
    assert client._llm is not None
    client.shutdown()
    assert client._llm is None
