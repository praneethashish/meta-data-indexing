import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_isbn_lookup():
    mock = AsyncMock()
    mock.return_value = {
        "title": "Enriched Title",
        "author": "Enriched Author",
        "publisher": "Enriched Publisher",
        "published_date": "2023-01-01",
    }
    return mock


@pytest.fixture
def sample_book_text():
    return """
    Title: Introduction to Python
    Author: John Smith
    Publisher: Tech Books Inc.
    ISBN: 978-0123456789
    Published: 2023
    """


@pytest.fixture
def sample_vparse_response():
    return {
        "results": {
            "test_book": {"md_content": "# Test Book\nBy Test Author\nISBN: 978-0123456789", "content_list": []}
        }
    }


@pytest.fixture
def sample_vparse_content_list():
    return {
        "results": {
            "test_book": {
                "md_content": "",
                "content_list": [
                    {"text": "Title: Test Book"},
                    {"text": "Author: Test Author"},
                    {"text": "ISBN: 978-0123456789"},
                ],
            }
        }
    }


@pytest.fixture
def sample_image_metadata():
    return {
        "width": 1920,
        "height": 1080,
        "format": "JPEG",
        "color_space": "RGB",
        "bit_depth": 24,
        "exif_camera_make": "Canon",
        "exif_camera_model": "EOS R5",
        "exif_date_taken": "2023:10:27 10:00:00",
        "exif_gps_latitude": 34.0522,
        "exif_gps_longitude": -118.2437,
        "exif_lens": "RF 24-70mm",
        "dpi_horizontal": 300.0,
        "dpi_vertical": 300.0,
    }


@pytest.mark.asyncio
async def test_process_pdf_calls_vparse(pipeline, sample_vparse_response, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    with (
        patch("bookextractor.pipeline.parse_pdf_via_vparse", new_callable=AsyncMock) as mock_vparse,
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=["978-0123456789"]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock) as mock_lookup,
    ):
        mock_vparse.return_value = sample_vparse_response
        mock_lookup.return_value = {}

        result = await pipeline.process_pdf(str(pdf_path))

        mock_vparse.assert_called_once()
        assert "book_metadata" in result


@pytest.mark.asyncio
async def test_process_pdf_fallback_content_list(pipeline, sample_vparse_content_list, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with (
        patch("bookextractor.pipeline.parse_pdf_via_vparse", new_callable=AsyncMock) as mock_vparse,
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=[]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock, return_value={}),
    ):
        mock_vparse.return_value = sample_vparse_content_list

        result = await pipeline.process_pdf(str(pdf_path))

        mock_vparse.assert_called_once()
        assert "book_metadata" in result


@pytest.mark.asyncio
async def test_process_image_extracts_metadata(pipeline, sample_image_metadata, tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    pipeline.client = None

    with patch("bookextractor.pipeline.extract_image_metadata", return_value=sample_image_metadata):
        result = await pipeline.process_image(str(image_path))

        assert "image_metadata" in result
        assert result["image_metadata"]["width"] == 1920
        assert result["image_metadata"]["height"] == 1080
        assert result["image_metadata"]["exif_camera_make"] == "Canon"


@pytest.mark.asyncio
async def test_process_image_benchmark_mode(pipeline, sample_image_metadata, tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0")

    pipeline.client = None

    with patch("bookextractor.pipeline.extract_image_metadata", return_value=sample_image_metadata):
        result = await pipeline.process_image(str(image_path), benchmark=True)

        assert "result" in result
        assert "debug" in result


@pytest.mark.asyncio
async def test_process_text_file_reads_content(pipeline, tmp_path):
    text_path = tmp_path / "test.md"
    text_path.write_text("# Test Book\nBy Test Author")

    with (
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=[]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock, return_value={}),
    ):
        result = await pipeline.process_text_file(str(text_path))

        assert "book_metadata" in result


@pytest.mark.asyncio
async def test_extract_from_text_with_isbn(pipeline, mock_isbn_lookup, sample_book_text):
    with (
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=["978-0123456789"]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock, side_effect=mock_isbn_lookup),
    ):
        result = await pipeline.extract_from_text(sample_book_text)

        assert "book_metadata" in result
        metadata = result["book_metadata"]
        assert metadata["isbn"] == "978-0123456789"


@pytest.mark.asyncio
async def test_extract_from_text_without_isbn(pipeline, sample_book_text):
    with (
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=[]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock, return_value={}),
    ):
        result = await pipeline.extract_from_text(sample_book_text)

        assert "book_metadata" in result
        metadata = result["book_metadata"]
        assert metadata["isbn"] is None


@pytest.mark.asyncio
async def test_extract_from_text_benchmark_mode(pipeline, sample_book_text):
    with (
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=[]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock, return_value={}),
    ):
        result = await pipeline.extract_from_text(sample_book_text, benchmark=True)

        assert "result" in result
        assert "debug" in result
        assert "isbn_candidates" in result["debug"]
        assert "llm_raw_output" in result["debug"]
        assert "text_snippet" in result["debug"]


@patch("bookextractor.model_client.ModelClient.get_instance")
def test_pipeline_init(mock_client):
    from bookextractor.pipeline import ExtractionPipeline

    _ = ExtractionPipeline(model_id="custom-model")
    mock_client.assert_called_with(model_id="custom-model", max_model_len=4096)


@pytest.mark.asyncio
async def test_process_pdf_invalid_json_fallback(pipeline, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    bad_vparse_response = {"results": {"test": {"content_list": "invalid { json", "md_content": "fallback text"}}}

    with (
        patch("bookextractor.pipeline.parse_pdf_via_vparse", new_callable=AsyncMock) as mock_vparse,
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=[]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock, return_value={}),
    ):
        mock_vparse.return_value = bad_vparse_response
        result = await pipeline.process_pdf(str(pdf_path))
        assert "book_metadata" in result


def test_calculate_confidence_with_data():
    from bookextractor.confidence_scorer import calculate_book_confidence

    final = {"title": "Test Book", "author": "Test Author", "publisher": "Test Publisher", "published_date": "2023"}

    candidates = {
        "title": ["Test Book", "Test Book", "Other Title"],
        "author": ["Test Author"],
        "publisher": ["Test Publisher", "Other Publisher"],
        "published_date": ["2023", "2023", "2023"],
    }

    scores = calculate_book_confidence(final, candidates, has_isbn=True)

    assert scores.title > 0.5
    assert scores.author == 1.0
    assert 0.5 <= scores.publisher <= 1.0
    assert scores.published_date == 1.0
    assert scores.isbn == 1.0


def test_calculate_confidence_without_data():
    from bookextractor.confidence_scorer import calculate_book_confidence

    final = {"title": None, "author": None, "publisher": None, "published_date": None}

    candidates: dict = {"title": [], "author": [], "publisher": [], "published_date": []}

    scores = calculate_book_confidence(final, candidates, has_isbn=False)

    assert scores.title == 0.0
    assert scores.author == 0.0
    assert scores.publisher == 0.0
    assert scores.published_date == 0.0
    assert scores.isbn == 0.0


@pytest.mark.asyncio
async def test_process_pdf_benchmark_mode(pipeline, sample_vparse_response, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with (
        patch("bookextractor.pipeline.parse_pdf_via_vparse", new_callable=AsyncMock) as mock_vparse,
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=[]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock, return_value={}),
    ):
        mock_vparse.return_value = sample_vparse_response
        result = await pipeline.process_pdf(str(pdf_path), benchmark=True)
        assert "result" in result
        assert "debug" in result


@pytest.mark.asyncio
async def test_extract_from_text_magazine(pipeline):
    magazine_text = "చందమామ మాసపత్రిక ఆగస్టు 1948 సంచిక 2 ఖరీదు 0-6-0"

    pipeline.client.generate_and_extract.return_value = {
        "magazine_name": "చందమామ",
        "issue_date": "August 1948",
        "issue_number": "2",
        "price": "0-6-0",
    }

    result = await pipeline.extract_from_text(magazine_text)

    assert "magazine_metadata" in result
    assert result["magazine_metadata"]["magazine_name"] == "చందమామ"
    assert result["magazine_metadata"]["issue_date"] == "August 1948"
    assert result["magazine_metadata"]["issue_number"] == "2"
    assert result["magazine_metadata"]["price"] == "0-6-0"
    assert result["magazine_metadata"]["language"] == "en"


@pytest.mark.asyncio
async def test_extract_from_text_magazine_benchmark(pipeline):
    magazine_text = "ఆంధ్రజ్యోతి మాసపత్రిక"

    pipeline.client.generate_and_extract.return_value = {
        "magazine_name": "ఆంధ్రజ్యోతి",
    }

    result = await pipeline.extract_from_text(magazine_text, benchmark=True)

    assert "result" in result
    assert "debug" in result
    assert result["result"]["magazine_metadata"]["magazine_name"] == "ఆంధ్రజ్యోతి"


@pytest.mark.asyncio
async def test_extract_from_text_magazine_json_input(pipeline, tmp_path):
    json_input = tmp_path / "input.json"
    json_input.write_text(
        json.dumps(
            {
                "transcription": "చందమామ మాసపత్రిక ఆగస్టు 1948",
                "language": "te",
            }
        )
    )

    pipeline.client.generate_and_extract.return_value = {
        "magazine_name": "చందమామ",
        "issue_date": "August 1948",
    }

    with (
        patch("bookextractor.pipeline.extract_isbn_candidates", return_value=[]),
        patch("bookextractor.pipeline.lookup_isbn", new_callable=AsyncMock, return_value={}),
    ):
        result = await pipeline.process_text_file(str(json_input))

        assert "magazine_metadata" in result
        assert result["magazine_metadata"]["magazine_name"] == "చందమామ"


def test_detect_content_type():
    from bookextractor.content_detector import detect_content_type

    assert detect_content_type("మాసపత్రిక విషయాలు") == "magazine"
    assert detect_content_type("సంచిక 2 నంపుటి") == "magazine"
    assert detect_content_type("subscription monthly") == "magazine"
    assert detect_content_type("This is a normal book about Python") == "book"
