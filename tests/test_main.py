from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None  # type: ignore

from metaextractor.main import app, get_vision_pipeline


@pytest.fixture
def test_client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_pdf_bytes():
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def sample_jpg_bytes():
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


@pytest.fixture
def override_vision_pipeline(mock_pipeline):
    app.dependency_overrides[get_vision_pipeline] = lambda: mock_pipeline
    yield mock_pipeline
    app.dependency_overrides.clear()


def test_health_endpoint(test_client):
    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_ready" in response.json()


def test_extract_jpg_routes_correctly(test_client, override_vision_pipeline):
    sample_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    files = {"file": ("test.jpg", BytesIO(sample_jpg), "image/jpeg")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 200
    override_vision_pipeline.process_image.assert_called_once()


def test_extract_png_routes_correctly(test_client, override_vision_pipeline):
    sample_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    files = {"file": ("test.png", BytesIO(sample_jpg), "image/png")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 200
    override_vision_pipeline.process_image.assert_called_once()


def test_extract_webp_routes_correctly(test_client, override_vision_pipeline):
    sample_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    files = {"file": ("test.webp", BytesIO(sample_jpg), "image/webp")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 200
    override_vision_pipeline.process_image.assert_called_once()


def test_extract_tiff_routes_correctly(test_client, override_vision_pipeline):
    sample_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    files = {"file": ("test.tiff", BytesIO(sample_jpg), "image/tiff")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 200
    override_vision_pipeline.process_image.assert_called_once()


def test_extract_jpeg_routes_correctly(test_client, override_vision_pipeline):
    sample_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    files = {"file": ("test.jpeg", BytesIO(sample_jpg), "image/jpeg")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 200
    override_vision_pipeline.process_image.assert_called_once()


def test_extract_tif_routes_correctly(test_client, override_vision_pipeline):
    sample_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    files = {"file": ("test.tif", BytesIO(sample_jpg), "image/tiff")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 200
    override_vision_pipeline.process_image.assert_called_once()


def test_extract_rejects_pdf_with_400(test_client):
    sample_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    files = {"file": ("test.pdf", BytesIO(sample_pdf), "application/pdf")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 400
    assert "Synchronous extraction is only supported for images" in response.json()["detail"]


def test_extract_rejects_md_with_400(test_client):
    sample_md = b"# Test Document\n\nBy Test Author"
    files = {"file": ("test.md", BytesIO(sample_md), "text/markdown")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 400
    assert "Synchronous extraction is only supported for images" in response.json()["detail"]


def test_extract_rejects_json_with_400(test_client):
    sample_json = b'{"title": "Test", "author": "Author"}'
    files = {"file": ("test.json", BytesIO(sample_json), "application/json")}
    response = test_client.post("/extract", files=files)

    assert response.status_code == 400
    assert "Synchronous extraction is only supported for images" in response.json()["detail"]


def test_extract_no_filename_returns_400(test_client):
    files = {"file": (None, BytesIO(b"data"), "application/pdf")}
    response = test_client.post("/extract", files=files)

    assert response.status_code in [400, 422]


def test_cli_pdf_routing(tmp_path):
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.pdf"
    input_file.write_bytes(b"%PDF-1.4\n")
    output_file = tmp_path / "output.json"

    mock_pipeline = MagicMock()
    mock_pipeline.process_pdf = AsyncMock(return_value={"book_metadata": {"title": "Test"}})

    with patch("metaextractor.main.ExtractionPipeline", return_value=mock_pipeline):
        result = runner.invoke(cli_app, ["extract", str(input_file), str(output_file)])

        assert result.exit_code == 0
        mock_pipeline.process_pdf.assert_called_once()
        assert mock_pipeline.process_pdf.call_args.kwargs["lang"] == "en"


def test_cli_image_routing(tmp_path):
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.jpg"
    input_file.write_bytes(b"\xff\xd8\xff\xe0")
    output_file = tmp_path / "output.json"

    mock_pipeline = MagicMock()
    mock_pipeline.process_image = AsyncMock(return_value={"image_metadata": {"width": 100}})

    with patch("metaextractor.main.ExtractionPipeline", return_value=mock_pipeline):
        result = runner.invoke(cli_app, ["extract", str(input_file), str(output_file)])

        assert result.exit_code == 0
        mock_pipeline.process_image.assert_called_once()


def test_cli_text_file_routing(tmp_path):
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.md"
    input_file.write_text("# Test")
    output_file = tmp_path / "output.json"

    mock_pipeline = MagicMock()
    mock_pipeline.process_text_file = AsyncMock(return_value={"book_metadata": {"title": "Test"}})

    with patch("metaextractor.main.ExtractionPipeline", return_value=mock_pipeline):
        result = runner.invoke(cli_app, ["extract", str(input_file), str(output_file)])

        assert result.exit_code == 0
        mock_pipeline.process_text_file.assert_called_once()


def test_cli_missing_arguments():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()

    result = runner.invoke(cli_app, ["extract"])

    assert result.exit_code == 2


def test_cli_extract_command_pdf_with_lang(tmp_path):
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.pdf"
    input_file.write_bytes(b"%PDF-1.4\n")
    output_file = tmp_path / "output.json"

    mock_pipeline = MagicMock()
    mock_pipeline.process_pdf = AsyncMock(return_value={"book_metadata": {"title": "Test"}})

    with patch("metaextractor.main.ExtractionPipeline", return_value=mock_pipeline):
        result = runner.invoke(cli_app, ["extract", str(input_file), str(output_file), "--lang", "te"])

        assert result.exit_code == 0, f"Exit code was {result.exit_code}, output: {result.output}"
        mock_pipeline.process_pdf.assert_called_once()
        assert mock_pipeline.process_pdf.call_args.kwargs["lang"] == "te"


def test_get_vision_pipeline_returns_pipeline():
    from metaextractor.main import get_vision_pipeline

    get_vision_pipeline.cache_clear()
    with patch("metaextractor.main.ExtractionPipeline") as mock_class:
        mock_class.return_value = MagicMock()
        pipeline = get_vision_pipeline()
        mock_class.assert_called_once_with(load_vlm=False)
        assert pipeline is not None

    get_vision_pipeline.cache_clear()


def test_get_vision_pipeline_is_cached():
    from metaextractor.main import get_vision_pipeline

    get_vision_pipeline.cache_clear()
    with patch("metaextractor.main.ExtractionPipeline") as mock_class:
        mock_class.return_value = MagicMock(name="pipeline_instance")
        p1 = get_vision_pipeline()
        p2 = get_vision_pipeline()
        assert p1 is p2
        assert mock_class.call_count == 1

    get_vision_pipeline.cache_clear()


def test_cli_unsupported_file_type_hits_else(tmp_path):
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    input_file = tmp_path / "test.docx"
    input_file.write_text("content")
    output_file = tmp_path / "output.json"

    with patch("metaextractor.main.ExtractionPipeline"):
        result = runner.invoke(cli_app, ["extract", str(input_file), str(output_file)])
        assert result.exit_code == 1
        assert "Unsupported file type" in result.output


def test_cli_api_command():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()

    with patch("metaextractor.main.uvicorn.run") as mock_run:
        result = runner.invoke(cli_app, ["api", "--host", "127.0.0.1", "--port", "9001"])

        assert result.exit_code == 0, f"Exit code was {result.exit_code}, output: {result.output}"
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == app
        assert mock_run.call_args[1]["host"] == "127.0.0.1"
        assert mock_run.call_args[1]["port"] == 9001


def test_cli_api_mode():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(cli_app, ["api"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 8000


def test_model_list_shows_models():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with (
        patch("metaextractor.hardware.get_vllm_config") as mock_cfg,
        patch("metaextractor.models_registry.is_model_cached", return_value=False),
        patch("metaextractor.models_registry.get_cached_model_size", return_value=0),
    ):
        mock_cfg.return_value = {"detected_hardware": {"memory_gb": 16}}
        result = runner.invoke(cli_app, ["model", "list"])
        assert result.exit_code == 0
        assert "Qwen/Qwen2.5-VL-7B-Instruct" in result.output
        assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in result.output
        assert "google/gemma-4-31b-it" in result.output


def test_model_list_with_cached():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()

    def is_cached(m):
        return m == "Qwen/Qwen2.5-VL-7B-Instruct"

    with (
        patch("metaextractor.hardware.get_vllm_config") as mock_cfg,
        patch("metaextractor.models_registry.is_model_cached", side_effect=is_cached),
        patch("metaextractor.models_registry.get_cached_model_size", return_value=8 * 1024**3),
    ):
        mock_cfg.return_value = {"detected_hardware": {"memory_gb": 16}}
        result = runner.invoke(cli_app, ["model", "list"])
        assert result.exit_code == 0


def test_model_cache_empty():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with patch("metaextractor.models_registry.get_cached_models", return_value=[]):
        result = runner.invoke(cli_app, ["model", "cache"])
        assert result.exit_code == 0
        assert "No models cached" in result.output


def test_model_cache_with_data():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with (
        patch("metaextractor.models_registry.get_cached_models", return_value=["Qwen/Qwen2.5-VL-7B-Instruct"]),
        patch("metaextractor.models_registry.get_cached_model_size", return_value=8 * 1024**3),
    ):
        result = runner.invoke(cli_app, ["model", "cache"])
        assert result.exit_code == 0
        assert "Qwen/Qwen2.5-VL-7B-Instruct" in result.output


def test_model_remove_no_match():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with (
        patch("metaextractor.models_registry.find_model_by_query", return_value=[]),
        patch("metaextractor.models_registry.get_cached_models", return_value=[]),
    ):
        result = runner.invoke(cli_app, ["model", "remove", "nonexistent"])
        assert result.exit_code == 1
        assert "No models match" in result.output


def test_model_remove_single_match():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    mock_match = [{"id": "testorg/testmodel"}]
    with (
        patch("metaextractor.models_registry.find_model_by_query", return_value=mock_match),
        patch("metaextractor.models_registry.remove_model_from_cache", return_value=True),
    ):
        result = runner.invoke(cli_app, ["model", "remove", "testmodel"])
        assert result.exit_code == 0
        assert "Removed" in result.output


def test_model_remove_not_cached():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    mock_match = [{"id": "testorg/testmodel"}]
    with (
        patch("metaextractor.models_registry.find_model_by_query", return_value=mock_match),
        patch("metaextractor.models_registry.remove_model_from_cache", return_value=False),
    ):
        result = runner.invoke(cli_app, ["model", "remove", "testmodel"])
        assert result.exit_code == 0
        assert "Not cached" in result.output


def test_model_remove_multiple_matches():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    mock_matches = [{"id": "google/gemma-3-27b-it"}, {"id": "google/gemma-4-31b-it"}]
    with (
        patch("metaextractor.models_registry.find_model_by_query", return_value=mock_matches),
        patch("questionary.select") as mock_select,
        patch("metaextractor.models_registry.remove_model_from_cache", return_value=True),
    ):
        mock_select.return_value.ask.return_value = "google/gemma-3-27b-it"
        result = runner.invoke(cli_app, ["model", "remove", "gemma"])
        assert result.exit_code == 0
        assert "Removed" in result.output


@pytest.mark.skipif(huggingface_hub is None, reason="huggingface_hub not installed")
def test_model_download_with_selection():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with (
        patch("metaextractor.models_registry.is_model_cached", return_value=False),
        patch("questionary.checkbox") as mock_checkbox,
        patch("huggingface_hub.snapshot_download") as mock_download,
    ):
        mock_checkbox.return_value.ask.return_value = ["testorg/testmodel"]
        result = runner.invoke(cli_app, ["model", "download"])
        assert result.exit_code == 0
        mock_download.assert_called_once_with("testorg/testmodel")


@pytest.mark.skipif(huggingface_hub is None, reason="huggingface_hub not installed")
def test_model_download_already_cached():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with (
        patch("metaextractor.models_registry.is_model_cached", return_value=True),
        patch("questionary.checkbox") as mock_checkbox,
        patch("huggingface_hub.snapshot_download") as mock_download,
    ):
        mock_checkbox.return_value.ask.return_value = ["testorg/testmodel"]
        result = runner.invoke(cli_app, ["model", "download"])
        assert result.exit_code == 0
        mock_download.assert_not_called()


@pytest.mark.skipif(huggingface_hub is None, reason="huggingface_hub not installed")
def test_model_download_no_selection():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with (
        patch("questionary.checkbox") as mock_checkbox,
    ):
        mock_checkbox.return_value.ask.return_value = None
        result = runner.invoke(cli_app, ["model", "download"])
        assert result.exit_code == 0
        assert "No models selected" in result.output


def test_hardware_info_command():
    from typer.testing import CliRunner

    from metaextractor.main import cli_app

    runner = CliRunner()
    with patch("metaextractor.hardware.get_vllm_config") as mock_cfg:
        mock_cfg.return_value = {
            "detected_hardware": {
                "device": "cuda",
                "name": "Tesla T4",
                "count": 1,
                "memory_gb": 14.56,
                "precision_supported": ["fp16"],
            },
            "dtype": "float16",
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
        }
        result = runner.invoke(cli_app, ["hardware-info"])
        assert result.exit_code == 0
        assert "Tesla T4" in result.output
        assert "14.56" in result.output
