import pytest

from bookextractor.vparse_client import parse_pdf_via_vparse


@pytest.fixture
def sample_vparse_response():
    return {"results": {"test_document": {"md_content": "# Document Title\n\nContent here...", "content_list": []}}}


@pytest.fixture
def sample_vparse_content_list():
    return {
        "results": {
            "test_document": {
                "md_content": "",
                "content_list": [
                    {"type": "text", "text": "Title: Document"},
                    {"type": "text", "text": "Page 1 content"},
                ],
            }
        }
    }


@pytest.fixture
def sample_pdf_path(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
    return str(pdf_path)


@pytest.mark.asyncio
async def test_parse_pdf_success_md_content(httpx_mock, sample_vparse_response, sample_pdf_path):
    httpx_mock.add_response(url="http://localhost:8000/file_parse", json=sample_vparse_response)

    result = await parse_pdf_via_vparse(sample_pdf_path)

    assert result == sample_vparse_response
    assert httpx_mock.get_request().method == "POST"


@pytest.mark.asyncio
async def test_parse_pdf_fallback_content_list(httpx_mock, sample_vparse_content_list, sample_pdf_path):
    httpx_mock.add_response(url="http://localhost:8000/file_parse", json=sample_vparse_content_list)

    result = await parse_pdf_via_vparse(sample_pdf_path)

    assert result == sample_vparse_content_list
    assert result["results"]["test_document"]["md_content"] == ""
    assert len(result["results"]["test_document"]["content_list"]) == 2


@pytest.mark.asyncio
async def test_parse_pdf_content_list_string(httpx_mock, sample_pdf_path):
    response = {"results": {"test_document": {"md_content": "", "content_list": "Plain text content as string"}}}

    httpx_mock.add_response(url="http://localhost:8000/file_parse", json=response)

    result = await parse_pdf_via_vparse(sample_pdf_path)

    assert result["results"]["test_document"]["content_list"] == "Plain text content as string"


@pytest.mark.asyncio
async def test_parse_pdf_api_error(httpx_mock, sample_pdf_path):
    httpx_mock.add_response(url="http://localhost:8000/file_parse", status_code=500)

    with pytest.raises(RuntimeError, match="vParse API call failed"):
        await parse_pdf_via_vparse(sample_pdf_path)


@pytest.mark.asyncio
async def test_parse_pdf_timeout(httpx_mock, sample_pdf_path):
    httpx_mock.add_exception(Exception("Request timeout"))

    with pytest.raises(RuntimeError, match="vParse API call failed"):
        await parse_pdf_via_vparse(sample_pdf_path)


@pytest.mark.asyncio
async def test_parse_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        await parse_pdf_via_vparse("/nonexistent/path/file.pdf")


@pytest.mark.asyncio
async def test_parse_pdf_correct_request_params(httpx_mock, sample_vparse_response, sample_pdf_path):
    httpx_mock.add_response(url="http://localhost:8000/file_parse", json=sample_vparse_response)

    await parse_pdf_via_vparse(sample_pdf_path)

    request = httpx_mock.get_request()
    assert request.method == "POST"
    assert request.url == "http://localhost:8000/file_parse"

    # Check form data
    data = request.read().decode("utf-8", errors="ignore")
    assert "pipeline" in data
    assert "test.pdf" in data
