import re


def validate_isbn(isbn: str) -> bool:
    # Remove hyphens and spaces
    isbn = re.sub(r"[\s\-]", "", isbn)

    if len(isbn) == 10:
        if not re.match(r"^\d{9}[\dX]$", isbn):
            return False
        total = 0
        for i in range(9):
            total += int(isbn[i]) * (10 - i)
        last = 10 if isbn[9] == "X" else int(isbn[9])
        total += last
        return total % 11 == 0
    elif len(isbn) == 13:
        if not re.match(r"^\d{13}$", isbn):
            return False
        total = 0
        for i in range(13):
            factor = 1 if i % 2 == 0 else 3
            total += int(isbn[i]) * factor
        return total % 10 == 0
    return False


def extract_isbn_candidates(text: str) -> list:
    # Regex for ISBN-10 and ISBN-13
    pattern = r"(?:\bISBN(?:-1[03])?:?\s*)?([0-9Xx\-\s]{10,20})"
    matches = re.findall(pattern, text)
    valid_isbns = []
    for m in matches:
        clean = re.sub(r"[\s\-]", "", m)
        # Check if it looks like an ISBN
        if len(clean) in [10, 13]:
            if validate_isbn(clean):
                valid_isbns.append(clean)
    return valid_isbns
