from bookextractor.config import settings
from bookextractor.confidence_scorer import (
    calculate_book_confidence,
    calculate_magazine_confidence,
)
from bookextractor.models import ConfidenceScores, MagazineConfidenceScores


def test_book_confidence_with_full_data():
    final = {"title": "Test Book", "author": "Test Author", "publisher": "Test Publisher", "published_date": "2023"}
    candidates = {
        "title": ["Test Book", "Test Book", "Other Title"],
        "author": ["Test Author"],
        "publisher": ["Test Publisher", "Other Publisher"],
        "published_date": ["2023", "2023", "2023"],
    }
    scores = calculate_book_confidence(final, candidates, has_isbn=True)

    assert isinstance(scores, ConfidenceScores)
    assert scores.title > 0.5
    assert scores.author == 1.0
    assert 0.5 <= scores.publisher <= 1.0
    assert scores.published_date == 1.0
    assert scores.isbn == 1.0


def test_book_confidence_without_data():
    final = {"title": None, "author": None, "publisher": None, "published_date": None}
    candidates = {"title": [], "author": [], "publisher": [], "published_date": []}
    scores = calculate_book_confidence(final, candidates, has_isbn=False)

    assert scores.title == 0.0
    assert scores.author == 0.0
    assert scores.publisher == 0.0
    assert scores.published_date == 0.0
    assert scores.isbn == 0.0


def test_book_confidence_partial_data():
    final = {"title": "Book", "author": None, "publisher": "Pub", "published_date": "2023"}
    candidates = {
        "title": ["Book"],
        "author": [],
        "publisher": ["Pub", "Other"],
        "published_date": ["2023"],
    }
    scores = calculate_book_confidence(final, candidates, has_isbn=False)

    assert scores.title > 0.0
    assert scores.author == 0.0
    assert scores.publisher > 0.0
    assert scores.published_date > 0.0
    assert scores.isbn == 0.0


def test_magazine_confidence_with_data():
    llm_result = {
        "magazine_name": "చందమామ",
        "editor": "చక్రపాతి",
        "publisher": "Pub",
        "issue_date": "August 1948",
        "issue_number": "2",
        "price": "0-6-0",
    }
    scores = calculate_magazine_confidence(llm_result)

    assert isinstance(scores, MagazineConfidenceScores)
    assert scores.magazine_name == 0.8
    assert scores.editor == 0.7
    assert scores.publisher == 0.7
    assert scores.issue_date == 0.8
    assert scores.issue_number == 0.6
    assert scores.price == 0.6


def test_magazine_confidence_empty_data():
    scores = calculate_magazine_confidence({})

    assert scores.magazine_name == 0.0
    assert scores.editor == 0.0
    assert scores.publisher == 0.0
    assert scores.issue_date == 0.0
    assert scores.issue_number == 0.0
    assert scores.price == 0.0


def test_magazine_confidence_partial_data():
    llm_result = {"magazine_name": "ఆంధ్రజ్యోతి", "price": "5"}
    scores = calculate_magazine_confidence(llm_result)

    assert scores.magazine_name == 0.8
    assert scores.editor == 0.0
    assert scores.publisher == 0.0
    assert scores.issue_date == 0.0
    assert scores.issue_number == 0.0
    assert scores.price == 0.6


def test_book_confidence_fields_constant():
    assert "title" in settings.BOOK_CONFIDENCE_FIELDS
    assert "author" in settings.BOOK_CONFIDENCE_FIELDS
    assert "publisher" in settings.BOOK_CONFIDENCE_FIELDS
    assert "published_date" in settings.BOOK_CONFIDENCE_FIELDS


def test_magazine_confidence_thresholds():
    assert "magazine_name" in settings.MAGAZINE_CONFIDENCE_THRESHOLDS
    assert "editor" in settings.MAGAZINE_CONFIDENCE_THRESHOLDS
    assert "publisher" in settings.MAGAZINE_CONFIDENCE_THRESHOLDS
    assert "issue_date" in settings.MAGAZINE_CONFIDENCE_THRESHOLDS
    assert "issue_number" in settings.MAGAZINE_CONFIDENCE_THRESHOLDS
    assert "price" in settings.MAGAZINE_CONFIDENCE_THRESHOLDS