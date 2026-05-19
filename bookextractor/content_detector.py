from .config import settings


def detect_content_type(text: str) -> str:
    """Detect if content is a magazine/periodical or book.

    Returns "magazine" if any magazine indicator pattern is found
    in the text (case-insensitive), otherwise returns "book".
    """
    text_lower = text.lower()
    for pattern in settings.MAGAZINE_PATTERNS:
        if pattern in text_lower:
            return "magazine"
    return "book"