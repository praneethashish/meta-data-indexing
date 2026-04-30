"""Vision Language Model (VLM) client for image description.

This module is a placeholder for future Phase 2 Image Level 2 VLM enhancement.
Currently, images only extract EXIF metadata via image_utils.

TODO: Implement Gemma-4 vLLM vision support for image description.
"""

import os


class VLMClient:
    """Client for Gemma-4 vLLM vision inference."""

    def __init__(self, model_id: str | None = None):
        raise NotImplementedError(
            "Image VLM description is not yet implemented. "
            "Only EXIF metadata extraction is available for images."
        )

    async def describe_image(self, image_path: str) -> dict:
        """Generate description for an image using Gemma-4 vLLM."""
        raise NotImplementedError(
            "Image VLM description is not yet implemented."
        )
