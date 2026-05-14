import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
)
async def lookup_isbn(isbn: str) -> dict[str, str | None]:
    """
    Lookup book metadata via Open Library API.
    """
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            key = f"ISBN:{isbn}"
            if key in data:
                book_info = data[key]
                authors = book_info.get("authors", [])
                publishers = book_info.get("publishers", [])
                return {
                    "title": book_info.get("title"),
                    "author": (", ".join([a["name"] for a in authors]) if authors else None),
                    "publisher": (", ".join([p["name"] for p in publishers]) if publishers else None),
                    "published_date": book_info.get("publish_date"),
                }
        except httpx.HTTPStatusError as e:
            logger.warning(f"Open Library HTTP {e.response.status_code} for ISBN {isbn}")
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.error(f"Open Library connection error for ISBN {isbn}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ISBN lookup for {isbn}: {e}")

    return {}
