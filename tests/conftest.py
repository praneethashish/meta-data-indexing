from unittest.mock import MagicMock, patch

import pytest

from bookextractor.model_manager import ModelManager
from bookextractor.pipeline import ExtractionPipeline


@pytest.fixture
def pipeline():
    mock_llm_client = MagicMock()
    mock_llm_client.extract_semantic_fields.return_value = {
        "title": "Test Book",
        "author": "Test Author",
        "publisher": "Test Publisher",
        "published_date": "2023",
    }
    mock_llm_client.extract_magazine_fields.return_value = {
        "magazine_name": "Test Magazine",
        "editor": "Test Editor",
    }

    mock_vision_client = MagicMock()

    with (
        patch("bookextractor.pipeline.create_llm_client", return_value=mock_llm_client),
        patch("bookextractor.pipeline.ModelClient.get_instance", return_value=mock_vision_client),
    ):
        p = ExtractionPipeline()
        p.llm_client = mock_llm_client
        p.vision_client = mock_vision_client
        return p


@pytest.fixture(autouse=True)
def _reset_model_manager():
    ModelManager.reset()
