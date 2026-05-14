import pytest

from bookextractor.external_api import lookup_isbn


@pytest.fixture
def sample_isbn_response():
    return {
        "ISBN:9780123456789": {
            "title": "The Great Gatsby",
            "authors": [{"name": "F. Scott Fitzgerald"}],
            "publishers": [{"name": "Scribner"}],
            "publish_date": "1925",
        }
    }


@pytest.fixture
def sample_empty_response():
    return {}


@pytest.fixture
def sample_partial_response():
    return {"ISBN:9780123456789": {"title": "Unknown Title", "authors": [], "publishers": [], "publish_date": None}}


@pytest.mark.asyncio
async def test_lookup_isbn_success(httpx_mock, sample_isbn_response):
    httpx_mock.add_response(
        url="https://openlibrary.org/api/books?bibkeys=ISBN:9780123456789&format=json&jscmd=data",
        json=sample_isbn_response,
    )

    result = await lookup_isbn("9780123456789")

    assert result["title"] == "The Great Gatsby"
    assert result["author"] == "F. Scott Fitzgerald"
    assert result["publisher"] == "Scribner"
    assert result["published_date"] == "1925"


@pytest.mark.asyncio
async def test_lookup_isbn_no_data(httpx_mock, sample_empty_response):
    httpx_mock.add_response(
        url="https://openlibrary.org/api/books?bibkeys=ISBN:9780000000000&format=json&jscmd=data",
        json=sample_empty_response,
    )

    result = await lookup_isbn("9780000000000")

    assert result == {}


@pytest.mark.asyncio
async def test_lookup_isbn_partial_data(httpx_mock, sample_partial_response):
    httpx_mock.add_response(
        url="https://openlibrary.org/api/books?bibkeys=ISBN:9780123456789&format=json&jscmd=data",
        json=sample_partial_response,
    )

    result = await lookup_isbn("9780123456789")

    assert result["title"] == "Unknown Title"
    assert result["author"] is None
    assert result["publisher"] is None
    assert result["published_date"] is None


@pytest.mark.asyncio
async def test_lookup_isbn_http_error(httpx_mock):
    httpx_mock.add_response(
        url="https://openlibrary.org/api/books?bibkeys=ISBN:9780123456789&format=json&jscmd=data", status_code=404
    )

    result = await lookup_isbn("9780123456789")

    assert result == {}


@pytest.mark.asyncio
async def test_lookup_isbn_multiple_authors(httpx_mock):
    response = {
        "ISBN:9780123456789": {
            "title": "Collaborative Work",
            "authors": [{"name": "Author One"}, {"name": "Author Two"}, {"name": "Author Three"}],
            "publishers": [{"name": "Academic Press"}],
            "publish_date": "2020",
        }
    }

    httpx_mock.add_response(
        url="https://openlibrary.org/api/books?bibkeys=ISBN:9780123456789&format=json&jscmd=data", json=response
    )

    result = await lookup_isbn("9780123456789")

    assert result["author"] == "Author One, Author Two, Author Three"


@pytest.mark.asyncio
async def test_lookup_isbn_multiple_publishers(httpx_mock):
    response = {
        "ISBN:9780123456789": {
            "title": "Joint Publication",
            "authors": [{"name": "Single Author"}],
            "publishers": [{"name": "Publisher A"}, {"name": "Publisher B"}],
            "publish_date": "2021",
        }
    }

    httpx_mock.add_response(
        url="https://openlibrary.org/api/books?bibkeys=ISBN:9780123456789&format=json&jscmd=data", json=response
    )

    result = await lookup_isbn("9780123456789")

    assert result["publisher"] == "Publisher A, Publisher B"
