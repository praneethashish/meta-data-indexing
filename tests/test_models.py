import pytest
from pydantic import ValidationError

from bookextractor.models import (
    BenchmarkResult,
    BookMetadata,
    ConfidenceScores,
    ExtractionResult,
    ImageMetadata,
    ImageVLMMetadata,
)


def test_confidence_scores_valid():
    """Test valid confidence scores within [0.0, 1.0]."""
    scores = ConfidenceScores(title=0.5, author=1.0, publisher=0.0, isbn=0.123, published_date=0.99)
    assert scores.title == 0.5
    assert scores.author == 1.0
    assert scores.publisher == 0.0


def test_confidence_scores_invalid_low():
    """Assert error for scores below 0.0."""
    with pytest.raises(ValidationError):
        ConfidenceScores(title=-0.1)


def test_confidence_scores_invalid_high():
    """Assert error for scores above 1.0."""
    with pytest.raises(ValidationError):
        ConfidenceScores(author=1.1)


def test_book_metadata_instantiation():
    """Test BookMetadata handles valid nested ConfidenceScores."""
    scores = ConfidenceScores(title=0.8)
    book = BookMetadata(title="Test Book", confidence=scores)
    assert book.title == "Test Book"
    assert book.confidence.title == 0.8
    assert book.author is None


def test_image_metadata_instantiation():
    """Test ImageMetadata requirements and optional fields."""
    # Valid mandatory fields
    img = ImageMetadata(width=1920, height=1080, format="JPEG")
    assert img.width == 1920
    assert img.exif_camera_make is None

    # Optional fields
    img_full = ImageMetadata(width=100, height=100, format="PNG", exif_gps_latitude=12.34, exif_gps_longitude=56.78)
    assert img_full.exif_gps_latitude == 12.34


def test_image_vlm_metadata_instantiation():
    """Test ImageVLMMetadata with entities list."""
    vlm = ImageVLMMetadata(description="A test scene", entities=[{"name": "object1"}])
    assert vlm.description == "A test scene"
    assert len(vlm.entities) == 1


def test_extraction_result_instantiation():
    """Test ExtractionResult can hold various metadata types."""
    res = ExtractionResult(book_metadata=BookMetadata(confidence=ConfidenceScores()))
    assert res.book_metadata is not None
    assert res.image_metadata is None


def test_benchmark_result_instantiation():
    """Test BenchmarkResult with extraction result and debug info."""
    result = ExtractionResult(image_metadata=ImageMetadata(width=10, height=10, format="BMP"))
    bench = BenchmarkResult(result=result, debug={"step1": "ok"})
    assert bench.result.image_metadata.width == 10
    assert bench.debug["step1"] == "ok"
