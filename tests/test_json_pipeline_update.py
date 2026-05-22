import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_process_json_with_transcription(pipeline):
    # Mock LLM generation through text backend
    pipeline.llm_client.extract_semantic_fields = MagicMock(
        return_value={"title": "Chandamama", "author": "Chakrapani"}
    )

    # Mock ISBN lookup
    with patch("metaextractor.pipeline.lookup_isbn", new_callable=AsyncMock) as mock_isbn:
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
    # Mock LLM generation through text backend
    pipeline.llm_client.extract_semantic_fields = MagicMock(return_value={"title": "Raw JSON Title"})

    with patch("metaextractor.pipeline.lookup_isbn", new_callable=AsyncMock) as mock_isbn:
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

    # Mock extract_semantic_fields to return a valid result and capture the text
    captured_text = {}

    def mock_extract_semantic_fields(text):
        captured_text["value"] = text
        return {"title": "Secret"}

    pipeline.llm_client.extract_semantic_fields = MagicMock(side_effect=mock_extract_semantic_fields)

    pipeline.extract_semantic_fields(long_text)

    # Check that the text contains more than 3000 characters
    assert "A" * 10000 in captured_text["value"]
    assert len(captured_text["value"]) > 10000
