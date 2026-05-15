import pytest

from bookextractor.content_detector import MAGAZINE_PATTERNS, detect_content_type


def test_detect_magazine_telugu_monthly():
    assert detect_content_type("మాసపత్రిక విషయాలు") == "magazine"


def test_detect_magazine_telugu_issue():
    assert detect_content_type("సంచిక 2 నంపుటి") == "magazine"


def test_detect_magazine_telugu_subscription():
    assert detect_content_type("చందా కట్టండి") == "magazine"


def test_detect_magazine_telugu_agents():
    assert detect_content_type("ఏజంట్లు కావాలి") == "magazine"


def test_detect_magazine_english():
    assert detect_content_type("This is a magazine about science") == "magazine"


def test_detect_magazine_issue():
    assert detect_content_type("Vol. 5, Issue 3") == "magazine"


def test_detect_magazine_no_dot():
    assert detect_content_type("No. 42 December issue") == "magazine"


def test_detect_magazine_subscription():
    assert detect_content_type("Take a subscription today") == "magazine"


def test_detect_magazine_monthly():
    assert detect_content_type("Our monthly newsletter") == "magazine"


def test_detect_magazine_periodical():
    assert detect_content_type("A periodical publication") == "magazine"


def test_detect_book_default():
    assert detect_content_type("This is a normal book about Python") == "book"


def test_detect_book_empty():
    assert detect_content_type("") == "book"


def test_detect_book_random_text():
    assert detect_content_type("The quick brown fox jumps over the lazy dog") == "book"


def test_magazine_patterns_is_list():
    assert isinstance(MAGAZINE_PATTERNS, list)
    assert len(MAGAZINE_PATTERNS) > 0