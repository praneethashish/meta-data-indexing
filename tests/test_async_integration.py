from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from bookextractor import celery_config
from bookextractor.main import app, cli_app
from bookextractor.tasks import extract_image_task, extract_pdf_task, extract_text_task, get_pipeline


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_extract_async_pdf(client):
    with patch("bookextractor.main.extract_pdf_task") as mock_task:
        mock_task.delay.return_value = MagicMock(id="test-job-id")

        pdf_content = b"%PDF-1.4\n..."
        files = {"file": ("test.pdf", pdf_content, "application/pdf")}
        response = client.post("/extract/async", files=files, data={"lang": "en"})

        assert response.status_code == 200
        assert response.json() == {"job_id": "test-job-id", "status": "submitted"}
        mock_task.delay.assert_called_once()


@pytest.mark.asyncio
async def test_extract_async_image(client):
    with patch("bookextractor.main.extract_image_task") as mock_task:
        mock_result = MagicMock(id="img-job-id")
        mock_task.apply_async.return_value = mock_result

        img_content = b"fake-image-data"
        files = {"file": ("test.jpg", img_content, "image/jpeg")}
        response = client.post("/extract/async", files=files)

        assert response.status_code == 200
        assert response.json() == {"job_id": "img-job-id", "status": "submitted"}
        mock_task.apply_async.assert_called_once()


@pytest.mark.asyncio
async def test_extract_async_text(client):
    with patch("bookextractor.main.extract_text_task") as mock_task:
        mock_task.delay.return_value = MagicMock(id="txt-job-id")

        txt_content = b"hello world"
        # Note: .txt is NOT in ALLOWED_EXTENSIONS, let's use .md
        files = {"file": ("test.md", txt_content, "text/markdown")}
        response = client.post("/extract/async", files=files)

        assert response.status_code == 200
        assert response.json() == {"job_id": "txt-job-id", "status": "submitted"}
        mock_task.delay.assert_called_once()


def test_get_job_status_pending(client):
    with patch("bookextractor.main.celery_app.AsyncResult") as mock_result:
        mock_instance = MagicMock()
        mock_instance.state = "PENDING"
        mock_instance.ready.return_value = False
        mock_result.return_value = mock_instance

        response = client.get("/jobs/test-id")
        assert response.status_code == 200
        assert response.json() == {"job_id": "test-id", "status": "PENDING", "ready": False}


def test_get_job_status_success(client):
    with patch("bookextractor.main.celery_app.AsyncResult") as mock_result:
        mock_instance = MagicMock()
        mock_instance.state = "SUCCESS"
        mock_instance.ready.return_value = True
        mock_instance.successful.return_value = True
        mock_instance.result = {"title": "Success"}
        mock_result.return_value = mock_instance

        response = client.get("/jobs/test-id")
        assert response.status_code == 200
        assert response.json()["status"] == "SUCCESS"
        assert response.json()["result"] == {"title": "Success"}


def test_get_job_status_failure(client):
    with patch("bookextractor.main.celery_app.AsyncResult") as mock_result:
        mock_instance = MagicMock()
        mock_instance.state = "FAILURE"
        mock_instance.ready.return_value = True
        mock_instance.successful.return_value = False
        mock_instance.result = "Task failed error"
        mock_result.return_value = mock_instance

        response = client.get("/jobs/test-id")
        assert response.status_code == 200
        assert response.json()["status"] == "FAILURE"
        assert response.json()["error"] == "Task failed error"


@patch("bookextractor.tasks.get_pipeline")
@patch("os.path.exists")
@patch("os.remove")
def test_tasks_pdf(mock_remove, mock_exists, mock_get_p):
    mock_pipeline = MagicMock()
    mock_pipeline.process_pdf_sync = MagicMock(return_value={"ok": True})
    mock_get_p.return_value = mock_pipeline
    mock_exists.return_value = True

    result = extract_pdf_task("fake.pdf")
    assert result == {"ok": True}
    mock_remove.assert_called_with("fake.pdf")


@patch("bookextractor.tasks.get_pipeline")
@patch("os.path.exists")
@patch("os.remove")
def test_tasks_image(mock_remove, mock_exists, mock_get_p):
    mock_pipeline = MagicMock()
    mock_pipeline.process_image_sync = MagicMock(return_value={"img": True})
    mock_get_p.return_value = mock_pipeline
    mock_exists.return_value = True

    result = extract_image_task("fake.jpg")
    assert result == {"img": True}
    mock_remove.assert_called_with("fake.jpg")


@patch("bookextractor.tasks.get_pipeline")
@patch("os.path.exists")
@patch("os.remove")
def test_tasks_text(mock_remove, mock_exists, mock_get_p):
    mock_pipeline = MagicMock()
    mock_pipeline.process_text_file_sync = MagicMock(return_value={"txt": True})
    mock_get_p.return_value = mock_pipeline
    mock_exists.return_value = True

    result = extract_text_task("fake.md")
    assert result == {"txt": True}
    mock_remove.assert_called_with("fake.md")


def test_cli_worker_command():
    runner = CliRunner()
    with patch("subprocess.run") as mock_run:
        # We need to mock subprocess.run because it's a blocking call
        # but the command should still be valid.
        result = runner.invoke(cli_app, ["worker", "--queue", "test_queue", "--concurrency", "1"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "celery" in args
        assert "test_queue" in args


def test_get_pipeline_lazy_loading():
    with patch("bookextractor.tasks.ExtractionPipeline") as mock_ep:
        # Clear global state for test
        import bookextractor.tasks as tasks

        tasks._pipeline_with_vlm = None
        tasks._pipeline_no_vlm = None

        # Test with LLM
        p1 = get_pipeline(load_vlm=True)
        mock_ep.assert_called_with(load_vlm=True)
        p2 = get_pipeline(load_vlm=True)
        assert p1 == p2
        assert mock_ep.call_count == 1

        # Test without LLM
        p3 = get_pipeline(load_vlm=False)
        mock_ep.assert_called_with(load_vlm=False)
        p4 = get_pipeline(load_vlm=False)
        assert p3 == p4
        assert mock_ep.call_count == 2


@pytest.mark.asyncio
async def test_extract_async_invalid_extension(client):
    files = {"file": ("test.exe", b"data", "application/octet-stream")}
    response = client.post("/extract/async", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_extract_async_no_filename(client):
    files = {"file": ("", b"data", "text/plain")}
    response = client.post("/extract/async", files=files)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_extract_async_submission_failure(client):
    with patch("bookextractor.main.extract_text_task.delay", side_effect=Exception("Submit error")):
        files = {"file": ("test.md", b"data", "text/markdown")}
        response = client.post("/extract/async", files=files)
        assert response.status_code == 500
        assert "Failed to submit task" in response.json()["detail"]


def test_celery_config_exists():
    assert celery_config.task_default_queue == "default_queue"
    assert "vlm_queue" in celery_config.task_queues
