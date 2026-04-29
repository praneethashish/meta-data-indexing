from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bookextractor.main import app


@pytest.fixture
def test_client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_pdf_bytes():
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def sample_jpg_bytes():
    # Minimal valid JPEG header
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"  # noqa: E501


@pytest.fixture
def sample_json_bytes():
    return b'{"title": "Test", "author": "Author"}'


@pytest.fixture
def sample_md_bytes():
    return b"# Test Document\n\nBy Test Author"


@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    mock.process_pdf = AsyncMock(return_value={"book_metadata": {"title": "PDF Book"}})
    mock.process_image = AsyncMock(return_value={"image_metadata": {"width": 100, "height": 100}})
    mock.process_text_file = AsyncMock(return_value={"book_metadata": {"title": "Text Book"}})
    return mock


def test_health_endpoint(test_client):
    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_extract_pdf_routes_correctly(test_client, sample_pdf_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_pdf.assert_called_once()


@pytest.mark.asyncio
async def test_extract_jpg_routes_correctly(test_client, sample_jpg_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.jpg", BytesIO(sample_jpg_bytes), "image/jpeg")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_image.assert_called_once()


@pytest.mark.asyncio
async def test_extract_png_routes_correctly(test_client, sample_jpg_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.png", BytesIO(sample_jpg_bytes), "image/png")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_image.assert_called_once()


@pytest.mark.asyncio
async def test_extract_webp_routes_correctly(test_client, sample_jpg_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.webp", BytesIO(sample_jpg_bytes), "image/webp")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_image.assert_called_once()


@pytest.mark.asyncio
async def test_extract_tiff_routes_correctly(test_client, sample_jpg_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.tiff", BytesIO(sample_jpg_bytes), "image/tiff")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_image.assert_called_once()


@pytest.mark.asyncio
async def test_extract_jpeg_routes_correctly(test_client, sample_jpg_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.jpeg", BytesIO(sample_jpg_bytes), "image/jpeg")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_image.assert_called_once()


@pytest.mark.asyncio
async def test_extract_md_routes_correctly(test_client, sample_md_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.md", BytesIO(sample_md_bytes), "text/markdown")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_text_file.assert_called_once()


@pytest.mark.asyncio
async def test_extract_json_routes_correctly(test_client, sample_json_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.json", BytesIO(sample_json_bytes), "application/json")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_text_file.assert_called_once()


@pytest.mark.asyncio
async def test_extract_unsupported_type_returns_400(test_client, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.txt", BytesIO(b"Plain text"), "text/plain")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
        mock_pipeline.process_pdf.assert_not_called()
        mock_pipeline.process_image.assert_not_called()
        mock_pipeline.process_text_file.assert_not_called()


@pytest.mark.asyncio
async def test_extract_no_filename_returns_400(test_client, sample_pdf_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("", BytesIO(sample_pdf_bytes), "application/pdf")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 422


@pytest.mark.asyncio
async def test_extract_uppercase_extension(test_client, sample_pdf_bytes, mock_pipeline):
    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        files = {"file": ("test.PDF", BytesIO(sample_pdf_bytes), "application/pdf")}
        response = test_client.post("/extract", files=files)

        assert response.status_code == 200
        mock_pipeline.process_pdf.assert_called_once()


def test_cli_pdf_routing(tmp_path):
    from typer.testing import CliRunner

    from bookextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.pdf"
    input_file.write_bytes(b"%PDF-1.4\n")
    output_file = tmp_path / "output.json"

    mock_pipeline = MagicMock()
    mock_pipeline.process_pdf = AsyncMock(return_value={"book_metadata": {"title": "Test"}})

    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        result = runner.invoke(cli_app, [str(input_file), str(output_file)])

        assert result.exit_code == 0
        mock_pipeline.process_pdf.assert_called_once()


def test_cli_image_routing(tmp_path):
    from typer.testing import CliRunner

    from bookextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.jpg"
    input_file.write_bytes(b"\xff\xd8\xff\xe0")
    output_file = tmp_path / "output.json"

    mock_pipeline = MagicMock()
    mock_pipeline.process_image = AsyncMock(return_value={"image_metadata": {"width": 100}})

    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        result = runner.invoke(cli_app, [str(input_file), str(output_file)])

        assert result.exit_code == 0
        mock_pipeline.process_image.assert_called_once()


def test_cli_text_file_routing(tmp_path):
    from typer.testing import CliRunner

    from bookextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.md"
    input_file.write_text("# Test")
    output_file = tmp_path / "output.json"

    mock_pipeline = MagicMock()
    mock_pipeline.process_text_file = AsyncMock(return_value={"book_metadata": {"title": "Test"}})

    with patch('bookextractor.main.get_pipeline', return_value=mock_pipeline):
        result = runner.invoke(cli_app, [str(input_file), str(output_file)])

        assert result.exit_code == 0
        mock_pipeline.process_text_file.assert_called_once()


def test_cli_missing_arguments():  # noqa: ARG001
    from typer.testing import CliRunner

    from bookextractor.main import cli_app

    runner = CliRunner()

    result = runner.invoke(cli_app, [])

    assert result.exit_code == 1
    assert "Missing arguments" in result.output


def test_cli_unsupported_file_type(tmp_path):
    from typer.testing import CliRunner

    from bookextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.txt"
    input_file.write_text("Plain text")
    output_file = tmp_path / "output.json"

    result = runner.invoke(cli_app, [str(input_file), str(output_file)])

    assert result.exit_code == 1
    assert "Unsupported file type" in result.output
