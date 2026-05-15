from unittest.mock import MagicMock, patch

import pytest

from bookextractor.model_manager import ModelManager
from bookextractor.pipeline import ExtractionPipeline
from bookextractor.vlm_client import VLMClient


@pytest.fixture
def pipeline():
    mock_generate_and_extract = {
        "title": "Test Book",
        "author": "Test Author",
        "publisher": "Test Publisher",
        "published_date": "2023",
    }

    mock_vlm_client = MagicMock()
    mock_text_backend = MagicMock()
    mock_text_backend.generate_and_extract.return_value = mock_generate_and_extract

    with patch.object(VLMClient, "get_instance", return_value=mock_vlm_client):
        p = ExtractionPipeline()
        p.vlm_client = mock_vlm_client
        p._text_backend = mock_text_backend
        return p


@pytest.fixture(autouse=True)
def _reset_model_manager():
    ModelManager.reset()
