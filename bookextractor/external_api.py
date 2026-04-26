import httpx


async def lookup_isbn(isbn: str) -> dict[str, str | None]:
    """
    Lookup book metadata via Open Library API.
    """
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
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

        except Exception:
            pass
    return {}
