MAGAZINE_PATTERNS = [
    "\u0c2e\u0c3e\u0c38\u0c2a\u0c24\u0c4d\u0c30\u0c3f\u0c15",
    "\u0c38\u0c02\u0c1a\u0c3f\u0c15",
    "\u0c1a\u0c02\u0c26\u0c3e",
    "\u0c0f\u0c1c\u0c02\u0c1f\u0c4d\u0c32\u0c41",
    "magazine",
    "issue",
    "vol.",
    "no.",
    "subscription",
    "monthly",
    "periodical",
]


def detect_content_type(text: str) -> str:
    """Detect if content is a magazine/periodical or book.

    Returns "magazine" if any magazine indicator pattern is found
    in the text (case-insensitive), otherwise returns "book".
    """
    text_lower = text.lower()
    for pattern in MAGAZINE_PATTERNS:
        if pattern in text_lower:
            return "magazine"
    return "book"