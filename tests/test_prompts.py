from bookextractor.config import settings
from bookextractor.prompts import (
    BOOK_EXTRACTION_PROMPT,
    DEFAULT_LANGUAGE_LABEL,
    IMAGE_ANALYSIS_PROMPT,
    LANGUAGE_LABELS,
    MAGAZINE_EXTRACTION_PROMPT,
)


def test_book_prompt_contains_expected_fields():
    assert '"title"' in BOOK_EXTRACTION_PROMPT
    assert '"author"' in BOOK_EXTRACTION_PROMPT
    assert '"publisher"' in BOOK_EXTRACTION_PROMPT
    assert '"published_date"' in BOOK_EXTRACTION_PROMPT


def test_magazine_prompt_contains_expected_fields():
    assert '"magazine_name"' in MAGAZINE_EXTRACTION_PROMPT
    assert '"editor"' in MAGAZINE_EXTRACTION_PROMPT
    assert '"publisher"' in MAGAZINE_EXTRACTION_PROMPT
    assert '"issue_date"' in MAGAZINE_EXTRACTION_PROMPT
    assert '"issue_number"' in MAGAZINE_EXTRACTION_PROMPT
    assert '"price"' in MAGAZINE_EXTRACTION_PROMPT


def test_image_prompt_contains_expected_keys():
    assert "description" in IMAGE_ANALYSIS_PROMPT
    assert "text_content" in IMAGE_ANALYSIS_PROMPT
    assert "language" in IMAGE_ANALYSIS_PROMPT
    assert "scene_classification" in IMAGE_ANALYSIS_PROMPT
    assert "entities" in IMAGE_ANALYSIS_PROMPT


def test_max_text_lengths_are_positive():
    assert settings.BOOK_EXTRACTION_MAX_LENGTH > 0
    assert settings.MAGAZINE_EXTRACTION_MAX_LENGTH > 0


def test_language_labels_has_expected_keys():
    assert "te" in LANGUAGE_LABELS
    assert "en" in LANGUAGE_LABELS
    assert "hi" in LANGUAGE_LABELS


def test_default_language_label():
    assert DEFAULT_LANGUAGE_LABEL == "Telugu (తెలుగు) and English"


def test_prompts_are_strings():
    assert isinstance(BOOK_EXTRACTION_PROMPT, str)
    assert isinstance(MAGAZINE_EXTRACTION_PROMPT, str)
    assert isinstance(IMAGE_ANALYSIS_PROMPT, str)


def test_book_prompt_has_format_placeholder():
    assert "{text}" in BOOK_EXTRACTION_PROMPT


def test_magazine_prompt_has_format_placeholders():
    assert "{lang_label}" in MAGAZINE_EXTRACTION_PROMPT
    assert "{text}" in MAGAZINE_EXTRACTION_PROMPT