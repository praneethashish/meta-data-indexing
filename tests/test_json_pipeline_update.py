import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bookextractor.pipeline import ExtractionPipeline


@pytest.fixture
def pipeline():
    # Mock vLLM to avoid loading actual model during tests
    with patch("bookextractor.pipeline.LLM"):
        return ExtractionPipeline()


@pytest.mark.asyncio
async def test_process_json_with_transcription(pipeline):
    # Mock LLM generation
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text='{"title": "Chandamama", "author": "Chakrapani"}')]
    pipeline.llm.generate = MagicMock(return_value=[mock_output])

    # Mock ISBN lookup
    with patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock) as mock_isbn:
        mock_isbn.return_value = {}

        # Create a temp JSON file with transcription and extra noise
        sample_data = {
            "transcription": "This is the real book text. Chandamama by Chakrapani.",
            "noise": "x" * 5000,
            "segments": [{"bbox": [1, 2, 3, 4]}],
        }

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
            json.dump(sample_data, tf)
            temp_path = tf.name

        try:
            # We want to verify that extract_semantic_fields receives ONLY the transcription
            with patch.object(
                pipeline, "extract_semantic_fields", wraps=pipeline.extract_semantic_fields
            ) as mock_extract:
                result = await pipeline.process_text_file(temp_path)

                # Verify LLM was called with the transcription, NOT the whole JSON
                mock_extract.assert_called_once()
                args, _ = mock_extract.call_args
                passed_text = args[0]

                assert passed_text == "This is the real book text. Chandamama by Chakrapani."
                assert "noise" not in passed_text
                assert "segments" not in passed_text

                assert result["book_metadata"]["title"] == "Chandamama"
                assert result["book_metadata"]["author"] == "Chakrapani"
        finally:
            os.remove(temp_path)


@pytest.mark.asyncio
async def test_process_json_fallback_if_no_transcription(pipeline):
    # Mock LLM generation
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text='{"title": "Raw JSON Title"}')]
    pipeline.llm.generate = MagicMock(return_value=[mock_output])

    with patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock) as mock_isbn:
        mock_isbn.return_value = {}

        # Create a temp JSON file WITHOUT transcription
        sample_data = {"title": "Raw JSON Title", "other": "data"}

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
            json.dump(sample_data, tf)
            temp_path = tf.name

        try:
            with patch.object(
                pipeline, "extract_semantic_fields", wraps=pipeline.extract_semantic_fields
            ) as mock_extract:
                await pipeline.process_text_file(temp_path)

                # Verify LLM was called with the stringified JSON as fallback
                mock_extract.assert_called_once()
                args, _ = mock_extract.call_args
                passed_text = args[0]

                assert '"title": "Raw JSON Title"' in passed_text
        finally:
            os.remove(temp_path)


@pytest.mark.asyncio
async def test_truncation_limit_increased(pipeline):
    # Create a very long string (over 3000 chars)
    long_text = "Book title is Secret. " + ("A" * 14000)

    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text='{"title": "Secret"}')]
    pipeline.llm.generate = MagicMock(return_value=[mock_output])

    # We test extract_semantic_fields directly to verify prompt construction
    with patch.object(pipeline.llm, "generate", return_value=[mock_output]) as mock_gen:
        pipeline.extract_semantic_fields(long_text)

        mock_gen.assert_called_once()
        args, _ = mock_gen.call_args
        prompt = args[0][0]

        # Check that the prompt contains more than 3000 characters of the text
        # (Actually we check if it contains the later part of the text)
        assert "A" * 10000 in prompt
        assert len(prompt) > 10000
