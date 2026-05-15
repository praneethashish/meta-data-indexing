import os
from unittest.mock import MagicMock, patch

import pytest

try:
    from vllm import SamplingParams as _SamplingParams
except ImportError:
    _SamplingParams = None

from bookextractor.model_manager import ModelManager
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
        patch("bookextractor.model_manager.LLM", MockLLM),
        patch("bookextractor.hardware.get_vllm_config", return_value=_MOCK_CONFIG),
    ):
        yield MockLLM


def _reset_vlm():
    VLMClient._instance = None
    ModelManager.reset()


@pytest.mark.skipif(_SamplingParams is None, reason="vllm not installed")
def test_vlm_client_singleton(mock_llm):  # noqa: ARG001
    _reset_vlm()
    client1 = VLMClient.get_instance()
    client2 = VLMClient.get_instance()

    assert client1 is client2
    assert client1._model_manager.is_loaded()

    _reset_vlm()


@pytest.mark.skipif(_SamplingParams is None, reason="vllm not installed")
@pytest.mark.asyncio
async def test_vlm_client_generate(mock_llm):  # noqa: ARG001
    _reset_vlm()
    client = VLMClient.get_instance()

    mock_instance = MagicMock()
    mock_output = MagicMock()
    mock_output.text = "generated text"
    mock_instance.generate.return_value = [MagicMock(outputs=[mock_output])]
    client._model_manager._model = mock_instance

    result = client.generate(["prompt"])
    assert result is not None
    mock_instance.generate.assert_called_once()

    custom_params = MagicMock()
    client.generate(["prompt"], sampling_params=custom_params)
    assert mock_instance.generate.call_count == 2

    _reset_vlm()


@pytest.mark.asyncio
async def test_vlm_client_describe_image(tmp_path):
    _reset_vlm()

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
    client._model_manager = ModelManager.get_instance()
    client._model_manager._model = mock_llm_instance
    client.model_id = "test-model"
    client.max_model_len = 4096

    with patch("bookextractor.vlm_client.Image") as mock_pil_image:
        mock_pil_image.open.return_value.convert.return_value = MagicMock()
        result = await client.describe_image(str(image_path))

    assert "description" in result
    assert result["description"] == "A test image"
    assert result["scene_classification"] == "document"

    _reset_vlm()


def test_vlm_client_generate_without_init():
    _reset_vlm()

    client = VLMClient.__new__(VLMClient)
    client._model_manager = ModelManager()
    client.model_id = "test-model"
    client.max_model_len = 4096

    with pytest.raises(RuntimeError, match="vLLM engine not initialized"):
        client.generate(["prompt"])


def test_vlm_client_delegates_model_detection_to_manager():
    _reset_vlm()

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

    _reset_vlm()


def test_vllm_target_device_env_guard(monkeypatch):
    _reset_vlm()

    monkeypatch.setenv("VLLM_TARGET_DEVICE", "custom_tpu_device")

    class SingeLLM:
        __name__ = "LLM"

        def __init__(self, **_kwargs):
            pass

    with (
        patch("bookextractor.hardware.get_vllm_config", return_value=_MOCK_CONFIG),
        patch("bookextractor.model_manager.LLM", SingeLLM),
    ):
        manager = ModelManager()
        manager.get_or_create(model_id="test-model", max_model_len=4096)

        assert os.environ["VLLM_TARGET_DEVICE"] == "custom_tpu_device"

    _reset_vlm()


def test_model_manager_thread_safety():
    _reset_vlm()

    manager = ModelManager.get_instance()
    assert manager.is_loaded() is False
    assert manager.get_model() is None

    class FakeLLM:
        pass

    manager._model = FakeLLM()
    assert manager.is_loaded() is True
    assert manager.get_model() is not None

    _reset_vlm()