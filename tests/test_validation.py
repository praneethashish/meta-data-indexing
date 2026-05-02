from bookextractor.validation import extract_isbn_candidates, validate_isbn


def test_validate_isbn10():
    assert validate_isbn("812601234X") is True
    assert validate_isbn("8126012345") is False


def test_validate_isbn13():
    assert validate_isbn("9788126012343") is True
    assert validate_isbn("9788126012345") is False


def test_validate_isbn_invalid_formats():
    # Invalid length (line 25)
    assert validate_isbn("123") is False
    # Invalid ISBN-10 format (contains non-X letters) (line 10)
    assert validate_isbn("812601234A") is False
    # Invalid ISBN-13 format (contains letters) (line 19)
    assert validate_isbn("978812601234A") is False


def test_extract_isbn_candidates():
    text = "The ISBN is 978-81-260-1234-3 and another one 812601234X"
    candidates = extract_isbn_candidates(text)
    assert "9788126012343" in candidates
    assert "812601234X" in candidates
