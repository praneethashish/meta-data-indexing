from unittest.mock import MagicMock, patch

import pytest

try:
    from vllm import SamplingParams as _SamplingParams
except ImportError:
    _SamplingParams = None  # type: ignore

from bookextractor.vlm_client import VLMClient


@pytest.fixture
def mock_llm():
    with patch("bookextractor.vlm_client.LLM") as mock:
        yield mock


@pytest.mark.skipif(_SamplingParams is None, reason="vllm not installed")
def test_vlm_client_singleton(mock_llm):
    """Test that VLMClient maintains a singleton instance."""
    # Reset singleton state for testing
    VLMClient._instance = None
    VLMClient._llm = None

    client1 = VLMClient.get_instance()
    client2 = VLMClient.get_instance()

    assert client1 is client2
    assert mock_llm.called


@pytest.mark.skipif(_SamplingParams is None, reason="vllm not installed")
@pytest.mark.asyncio
async def test_vlm_client_generate(mock_llm):
    """Test the generate method of VLMClient."""
    VLMClient._instance = None
    VLMClient._llm = None

    mock_instance = MagicMock()
    mock_instance.generate.return_value = ["output"]
    mock_llm.return_value = mock_instance

    client = VLMClient.get_instance()
    result = client.generate(["prompt"])

    assert result == ["output"]
    mock_instance.generate.assert_called_once()

    # Custom params
    custom_params = _SamplingParams(temperature=0.0)
    client.generate(["prompt"], sampling_params=custom_params)
    assert mock_instance.generate.call_count == 2


@pytest.mark.asyncio
async def test_vlm_client_describe_image():
    """Test the describe_image placeholder."""
    VLMClient._instance = None
    VLMClient._llm = None

    # Instantiate without actual LLM for simple placeholder test
    with patch("bookextractor.vlm_client.VLMClient._initialize_llm"):
        client = VLMClient.get_instance()
        result = await client.describe_image("test_path")

    assert "description" in result
    assert result["text_content"] is None


def test_vlm_client_generate_without_init():
    """Test that generate raises RuntimeError without LLM initialized."""
    VLMClient._instance = None
    VLMClient._llm = None

    client = VLMClient.__new__(VLMClient)
    client._llm = None

    with pytest.raises(RuntimeError, match="vLLM engine not initialized"):
        client.generate(["prompt"])


def test_vlm_client_auto_detect_cached():
    """Test auto-detection of cached models."""
    VLMClient._instance = None
    VLMClient._llm = None

    with patch("bookextractor.models_registry.get_cached_models", return_value=[]):
        result = VLMClient._detect_cached_model()
        assert result == "Qwen/Qwen2.5-VL-7B-Instruct"

    with patch("bookextractor.models_registry.get_cached_models", return_value=["myorg/mymodel"]):
        result = VLMClient._detect_cached_model()
        assert result == "myorg/mymodel"
