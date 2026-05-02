import pytest

from bookextractor.vlm_client import VLMClient


def test_vlm_client_init_raises_not_implemented():
    """Assert VLMClient.__init__ raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as excinfo:
        VLMClient()
    assert "Image VLM description is not yet implemented" in str(excinfo.value)


@pytest.mark.asyncio
async def test_vlm_client_describe_image_raises_not_implemented():
    """Assert VLMClient.describe_image raises NotImplementedError."""
    # We use a trick to instantiate without triggering __init__'s NotImplementedError
    client = VLMClient.__new__(VLMClient)

    with pytest.raises(NotImplementedError) as excinfo:
        await client.describe_image("test_image.jpg")
    assert "Image VLM description is not yet implemented" in str(excinfo.value)
