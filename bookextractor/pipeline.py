import asyncio
import json
import logging
import os
import re
from contextlib import suppress
from typing import Any

from .config import settings
from .content_detector import detect_content_type
from .confidence_scorer import calculate_book_confidence, calculate_magazine_confidence
from .exceptions import ModelNotAvailableError, ParsingError
from .external_api import lookup_isbn
from .image_utils import extract_image_metadata
from .model_client import ModelClient
from .models import (
    BenchmarkResult,
    BookMetadata,
    ExtractionResult,
    ImageMetadata,
    ImageVLMMetadata,
    MagazineMetadata,
)
from .prompts import (
    BOOK_EXTRACTION_PROMPT,
    DEFAULT_LANGUAGE_LABEL,
    LANGUAGE_LABELS,
    MAGAZINE_EXTRACTION_PROMPT,
)
from .validation import extract_isbn_candidates
from .vparse_client import parse_pdf_via_vparse

logger = logging.getLogger(__name__)



class ExtractionPipeline:
    def __init__(self, model_id: str | None = None, max_model_len: int = 4096, load_vlm: bool = True):
        self.client: ModelClient | None = None
        if load_vlm:
            self.client = ModelClient.get_instance(model_id=model_id, max_model_len=max_model_len)

    async def process_pdf(self, pdf_path: str, benchmark: bool = False, lang: str = "en") -> dict[str, Any]:
        # Call vParse OCR API with the selected language
        vparse_response = await parse_pdf_via_vparse(pdf_path, lang=lang)

        results = vparse_response.get("results", {})
        filename = os.path.basename(pdf_path).rsplit(".", 1)[0]
        result_data = results.get(filename, {})

        # Fallback: if filename key not found, use the first available result
        if not result_data and results:
            result_data = results[next(iter(results))]

        # Extract text, preferring content_list (cleaner) over md_content
        full_text = ""

        if isinstance(result_data, dict):
            content_list = result_data.get("content_list", [])
            if isinstance(content_list, str):
                with suppress(json.JSONDecodeError, TypeError):
                    content_list = json.loads(content_list)
            if isinstance(content_list, list):
                full_text = "\n".join(
                    item.get("text", "") for item in content_list if isinstance(item, dict) and item.get("text")
                )

        # Fallback to md_content
        if not full_text and isinstance(result_data, dict):
            full_text = result_data.get("md_content", "")

        # Clean up the text for LLM
        full_text = re.sub(r"!\[.*?\]\(.*?\)\s*", "", full_text)  # Remove image refs
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)  # Collapse whitespace
        full_text = full_text.strip()

        return await self.extract_from_text(full_text, benchmark=benchmark, lang=lang)

    async def process_image(self, image_path: str, benchmark: bool = False) -> dict[str, Any]:
        metadata_dict = extract_image_metadata(image_path)
        img_meta = ImageMetadata(**metadata_dict)

        image_vlm_metadata = None
        if self.client is not None:
            vlm_data_dict = await self.client.describe_image(image_path)
            image_vlm_metadata = ImageVLMMetadata(**vlm_data_dict)

        result = ExtractionResult(image_metadata=img_meta, image_vlm_metadata=image_vlm_metadata)

        if benchmark:
            return BenchmarkResult(result=result).model_dump()
        return result.model_dump()

    async def process_text_file(self, file_path: str, benchmark: bool = False, lang: str = "en") -> dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return ExtractionResult().model_dump()

        # Check if it's a structured JSON with transcription (already OCR'd content)
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "transcription" in data:
                transcription = data.get("transcription")
                if transcription is None:
                    transcription = json.dumps(data)
                detected_lang = data.get("language", lang)
                return await self.extract_from_text(str(transcription), benchmark=benchmark, lang=detected_lang)
        except (json.JSONDecodeError, TypeError):
            pass
        return await self.extract_from_text(content, benchmark=benchmark, lang=lang)

    async def extract_from_text(self, text: str, benchmark: bool = False, lang: str = "en") -> dict[str, Any]:
        content_type = detect_content_type(text)

        if content_type == "magazine":
            return await self._extract_magazine_metadata(text, benchmark=benchmark, lang=lang)
        else:
            return await self._extract_book_metadata(text, benchmark=benchmark)

    async def _extract_book_metadata(self, text: str, benchmark: bool = False) -> dict[str, Any]:
        # ISBN Extraction
        isbns = extract_isbn_candidates(text)

        # LLM Semantic Extraction
        try:
            llm_result = self.extract_semantic_fields(text)
        except (ModelNotAvailableError, ParsingError) as e:
            logger.warning(f"Book metadata extraction failed: {e}")
            llm_result = {}

        final_data = {
            "title": llm_result.get("title"),
            "author": llm_result.get("author"),
            "publisher": llm_result.get("publisher"),
            "published_date": llm_result.get("published_date"),
            "isbn": None,
        }

        # ISBN Validation & External lookup
        isbn = None
        if isbns:
            isbn = isbns[0]
            ext_data = await lookup_isbn(isbn)
            for k, v in ext_data.items():
                if v:
                    final_data[k] = v

        final_data["isbn"] = isbn

        # Confidence Scoring
        all_candidates: dict[str, list[Any]] = {k: [v] for k, v in final_data.items() if k != "isbn"}
        all_candidates["isbn"] = isbns
        confidence = calculate_book_confidence(final_data, all_candidates, bool(isbn))

        book_meta = BookMetadata(
            title=final_data.get("title"),
            author=final_data.get("author"),
            publisher=final_data.get("publisher"),
            isbn=final_data.get("isbn"),
            published_date=final_data.get("published_date"),
            confidence=confidence,
        )

        result = ExtractionResult(book_metadata=book_meta)
        debug_info = {"isbn_candidates": isbns, "llm_raw_output": llm_result, "text_snippet": text[:settings.TEXT_SNIPPET_LENGTH]}

        if benchmark:
            return BenchmarkResult(result=result, debug=debug_info).model_dump()
        return result.model_dump()

    async def _extract_magazine_metadata(self, text: str, benchmark: bool = False, lang: str = "te") -> dict[str, Any]:
        # LLM Semantic Extraction for magazines
        try:
            llm_result = self.extract_magazine_semantic_fields(text, lang=lang)
        except (ModelNotAvailableError, ParsingError) as e:
            logger.warning(f"Magazine metadata extraction failed: {e}")
            llm_result = {}

        # Build confidence scores
        confidence = calculate_magazine_confidence(llm_result)

        magazine_meta = MagazineMetadata(
            magazine_name=llm_result.get("magazine_name"),
            editor=llm_result.get("editor"),
            publisher=llm_result.get("publisher"),
            issue_date=llm_result.get("issue_date"),
            issue_number=llm_result.get("issue_number"),
            price=llm_result.get("price"),
            language=lang,
            confidence=confidence,
        )

        result = ExtractionResult(magazine_metadata=magazine_meta)
        debug_info = {"llm_raw_output": llm_result, "text_snippet": text[:settings.TEXT_SNIPPET_LENGTH]}

        if benchmark:
            return BenchmarkResult(result=result, debug=debug_info).model_dump()
        return result.model_dump()

    def extract_magazine_semantic_fields(self, text: str, lang: str = "te") -> dict[str, Any]:
        lang_label = LANGUAGE_LABELS.get(lang, DEFAULT_LANGUAGE_LABEL)
        prompt = MAGAZINE_EXTRACTION_PROMPT.format(lang_label=lang_label, text=text[:settings.MAGAZINE_EXTRACTION_MAX_LENGTH])
        if self.client is None:
            raise ModelNotAvailableError()
        result = self.client.generate_and_extract(prompt, temperature=0.1, max_tokens=512)
        if result is not None:
            logger.info(f"Raw LLM magazine output: {result}")
            return result
        raise ParsingError("Magazine extraction prompt did not produce valid JSON")

    def extract_semantic_fields(self, text: str) -> dict[str, Any]:
        prompt = BOOK_EXTRACTION_PROMPT.format(text=text[:settings.BOOK_EXTRACTION_MAX_LENGTH])
        if self.client is None:
            raise ModelNotAvailableError()
        result = self.client.generate_and_extract(prompt, max_tokens=256)
        if result is not None:
            return result
        raise ParsingError("Book extraction prompt did not produce valid JSON")

    def process_pdf_sync(self, pdf_path: str, benchmark: bool = False, lang: str = "en") -> dict[str, Any]:
        """Synchronous wrapper for process_pdf. Safe for Celery tasks."""
        return asyncio.run(self.process_pdf(pdf_path, benchmark=benchmark, lang=lang))

    def process_image_sync(self, image_path: str, benchmark: bool = False) -> dict[str, Any]:
        """Synchronous wrapper for process_image. Safe for Celery tasks."""
        return asyncio.run(self.process_image(image_path, benchmark=benchmark))

    def process_text_file_sync(self, file_path: str, benchmark: bool = False, lang: str = "en") -> dict[str, Any]:
        """Synchronous wrapper for process_text_file. Safe for Celery tasks."""
        return asyncio.run(self.process_text_file(file_path, benchmark=benchmark, lang=lang))

    
