from unittest.mock import MagicMock, patch

import pytest

from bookextractor.vlm_client import VLMClient


@pytest.fixture
def mock_llm():
    with patch("bookextractor.vlm_client.LLM") as mock:
        yield mock


def test_vlm_client_singleton(mock_llm):
    """Test that VLMClient maintains a singleton instance."""
    # Reset singleton state for testing
    VLMClient._instance = None
    VLMClient._llm = None

    client1 = VLMClient.get_instance()
    client2 = VLMClient.get_instance()

    assert client1 is client2
    assert mock_llm.called


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
    from vllm import SamplingParams

    custom_params = SamplingParams(temperature=0.0)
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
