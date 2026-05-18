from unittest.mock import MagicMock, patch

import pytest

from bookextractor.model_manager import ModelManager
from bookextractor.pipeline import ExtractionPipeline


@pytest.fixture
def pipeline():
    mock_client = MagicMock()
    mock_client.generate_and_extract.return_value = {
        "title": "Test Book",
        "author": "Test Author",
        "publisher": "Test Publisher",
        "published_date": "2023",
    }

    with patch("bookextractor.model_client.ModelClient.get_instance", return_value=mock_client):
        p = ExtractionPipeline()
        p.client = mock_client
        return p


@pytest.fixture(autouse=True)
def _reset_model_manager():
    ModelManager.reset()
