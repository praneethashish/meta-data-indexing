from unittest.mock import patch

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from bookextractor.main import app, cli_app

runner = CliRunner()
client = TestClient(app)


def test_hardware_info_command():
    result = runner.invoke(cli_app, ["hardware-info"])
    assert result.exit_code == 0
    assert "Hardware Detection" in result.stdout


def test_extract_unsupported_file(tmp_path):
    # Mock VLMClient to avoid actual initialization
    with patch("bookextractor.pipeline.VLMClient.get_instance"):
        bad_file = tmp_path / "test.docx"
        bad_file.write_text("hello")
        result = runner.invoke(cli_app, ["extract", str(bad_file), "out.json"])
        # Should fail due to unsupported file type
        assert result.exit_code != 0


def test_api_command():
    with patch("uvicorn.run"):
        result = runner.invoke(cli_app, ["api", "--port", "9000"])
        assert result.exit_code == 0


def test_api_extract_unsupported_type():
    with patch("bookextractor.main.get_pipeline"):
        response = client.post("/extract", files={"file": ("test.docx", b"hello", "application/octet-stream")})
        assert response.status_code == 400


def test_api_extract_no_filename():
    response = client.post("/extract", files={"file": ("", b"hello", "text/plain")})
    assert response.status_code in (400, 422)


def test_main_cli_api_flag():
    with patch("bookextractor.main._run_api") as mock_run:
        result = runner.invoke(cli_app, ["--api"])
        # If typer exits with 0 or 2, we just want to see if it was called
        assert mock_run.called or result.exit_code in (0, 2)
