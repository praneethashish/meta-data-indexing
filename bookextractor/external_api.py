import httpx
from typing import Dict, Optional

async def lookup_isbn(isbn: str) -> Dict[str, Optional[str]]:
    """
    Lookup book metadata via Open Library API.
    """
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                key = f"ISBN:{isbn}"
                if key in data:
                    book_info = data[key]
                    return {
                        "title": book_info.get("title"),
                        "author": ", ".join([a["name"] for a in book_info.get("authors", [])]) if book_info.get("authors") else None,
                        "publisher": ", ".join([p["name"] for p in book_info.get("publishers", [])]) if book_info.get("publishers") else None,
                        "published_date": book_info.get("publish_date")
                    }
        except Exception:
            pass
    return {}
