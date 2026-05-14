import json
import logging
import os
import re
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

try:
    from vllm import SamplingParams
except ImportError:

    class _SamplingParamsStub:
        def __init__(self, **kwargs: Any) -> None:
            pass

    SamplingParams = _SamplingParamsStub  # type: ignore

from .external_api import lookup_isbn
from .image_utils import extract_image_metadata
from .models import (
    BenchmarkResult,
    BookMetadata,
    ConfidenceScores,
    ExtractionResult,
    ImageMetadata,
    ImageVLMMetadata,
    MagazineConfidenceScores,
    MagazineMetadata,
)
from .validation import extract_isbn_candidates
from .vlm_client import VLMClient
from .vparse_client import parse_pdf_via_vparse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = Path(os.getenv("BOOKEXTRACTOR_MODELS_DIR", PROJECT_ROOT / "models"))


class ExtractionPipeline:
    def __init__(self, model_id: str | None = None, max_model_len: int = 4096, load_llm: bool = True):
        self.llm: Any = None
        self.vlm_client: VLMClient | None = None
        if load_llm:
            self.vlm_client = VLMClient.get_instance(model_id=model_id, max_model_len=max_model_len)
            self.llm = VLMClient._llm

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
        if self.vlm_client is not None:
            vlm_data_dict = await self.vlm_client.describe_image(image_path)
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

    def _detect_content_type(self, text: str) -> str:
        """Detect if content is a magazine/periodical or book."""
        text_lower = text.lower()

        # Magazine indicators (English and Telugu)
        magazine_patterns = [
            "మాసపత్రిక",  # monthly magazine (Telugu)
            "సంచిక",  # issue (Telugu)
            "చందా",  # subscription (Telugu)
            "ఏజంట్లు",  # agents (Telugu)
            "magazine",
            "issue",
            "vol.",
            "no.",
            "subscription",
            "monthly",
            "periodical",
        ]

        for pattern in magazine_patterns:
            if pattern in text_lower:
                return "magazine"

        return "book"

    async def extract_from_text(self, text: str, benchmark: bool = False, lang: str = "en") -> dict[str, Any]:
        content_type = self._detect_content_type(text)

        if content_type == "magazine":
            return await self._extract_magazine_metadata(text, benchmark=benchmark, lang=lang)
        else:
            return await self._extract_book_metadata(text, benchmark=benchmark)

    async def _extract_book_metadata(self, text: str, benchmark: bool = False) -> dict[str, Any]:
        # ISBN Extraction
        isbns = extract_isbn_candidates(text)

        # LLM Semantic Extraction
        llm_result = self.extract_semantic_fields(text)

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
        confidence = self.calculate_confidence(final_data, all_candidates, bool(isbn))

        book_meta = BookMetadata(
            title=final_data.get("title"),
            author=final_data.get("author"),
            publisher=final_data.get("publisher"),
            isbn=final_data.get("isbn"),
            published_date=final_data.get("published_date"),
            confidence=confidence,
        )

        result = ExtractionResult(book_metadata=book_meta)
        debug_info = {"isbn_candidates": isbns, "llm_raw_output": llm_result, "text_snippet": text[:500]}

        if benchmark:
            return BenchmarkResult(result=result, debug=debug_info).model_dump()
        return result.model_dump()

    async def _extract_magazine_metadata(self, text: str, benchmark: bool = False, lang: str = "te") -> dict[str, Any]:
        # LLM Semantic Extraction for magazines
        llm_result = self.extract_magazine_semantic_fields(text, lang=lang)

        # Build confidence scores
        confidence = MagazineConfidenceScores(
            magazine_name=0.8 if llm_result.get("magazine_name") else 0.0,
            editor=0.7 if llm_result.get("editor") else 0.0,
            publisher=0.7 if llm_result.get("publisher") else 0.0,
            issue_date=0.8 if llm_result.get("issue_date") else 0.0,
            issue_number=0.6 if llm_result.get("issue_number") else 0.0,
            price=0.6 if llm_result.get("price") else 0.0,
        )

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
        debug_info = {"llm_raw_output": llm_result, "text_snippet": text[:500]}

        if benchmark:
            return BenchmarkResult(result=result, debug=debug_info).model_dump()
        return result.model_dump()

    def extract_magazine_semantic_fields(self, text: str, lang: str = "te") -> dict[str, Any]:
        lang_label = {
            "te": "Telugu (తెలుగు)",
            "en": "English",
            "hi": "Hindi (हिन्दी)",
        }.get(lang, "Telugu (తెలుగు) and English")
        prompt = f"""Extract magazine metadata from the OCR text below.
The text is primarily in {lang_label}.

Return ONLY a valid JSON object with these fields (use null for unknown fields):
- "magazine_name": Name of the magazine
- "editor": Editor name
- "publisher": Publisher name
- "issue_date": Issue month and year
- "issue_number": Issue/volume number
- "price": Price

Example:
{{"magazine_name": "చందమామ", "editor": "చక్రపాతి", "issue_date": "August 1948", "issue_number": "2"}}

OCR Text:
{text[:3000]}

JSON:
"""
        if self.vlm_client is None:
            return {}
        text_out = ""
        try:
            if self.vlm_client is None:
                return {}
            sampling_params = SamplingParams(temperature=0.1, max_tokens=512, stop=["```"])
            outputs = self.vlm_client.generate([prompt], sampling_params)
            text_out = outputs[0].outputs[0].text.strip()
            start = text_out.find("{")
            end = text_out.rfind("}") + 1
            if start != -1 and end != -1:
                raw = text_out[start:end]
                logger.info(f"Raw LLM magazine output: {raw}")
                result = cast(dict[str, Any], json.loads(raw))
                return result
            else:
                logger.warning(f"No JSON found in LLM output: {text_out}")
        except Exception as e:
            logger.warning(f"Failed to parse magazine JSON: {e}, raw: {text_out}")
        return {}

    def extract_semantic_fields(self, text: str) -> dict[str, Any]:
        prompt = f"""You are an expert book metadata extractor. Your task is to identify and extract
structured metadata from noisy OCR text of scanned book pages.

The text below was extracted via OCR from scanned book pages and may contain:
- OCR errors, garbled characters, or misread words
- Mixed languages (English, Telugu, Hindi)
- Publishing information like edition details, print runs, and pricing
- Copyright notices with author names
- Publisher addresses and contact information

Extract the following fields and return ONLY a valid JSON object:
- "title": The book's title (look for prominent text, large headings, or text on the title page)
- "author": The author's full name (look near "By", "©", "Written by", or Telugu/Hindi equivalents)
- "publisher": The publisher's name (look near "Published by", "ప్రచురణ", "प्रकाशక", or publishing house names)
- "published_date": The earliest publication date (look for years like 1996, 2004 near
  "First Edition", "ముద్రణ", "संस्करण")

Rules:
- Return STRICT JSON only — no explanation, no markdown, no extra text.
- If a field cannot be confidently determined, set its value to null.
- Do NOT fabricate or guess values. Only extract what is clearly present.
- Do NOT include ISBN (it is extracted separately).
- Prefer the original/first edition date over reprint dates.

OCR Text:
{text[:15000]}

JSON:
"""
        try:
            if self.vlm_client is None:
                return {}
            sampling_params = SamplingParams(temperature=0.7, max_tokens=256, stop=["```"])
            outputs = self.vlm_client.generate([prompt], sampling_params)
            text_out = outputs[0].outputs[0].text.strip()
            start = text_out.find("{")
            end = text_out.rfind("}") + 1
            if start != -1 and end != -1:
                return cast(dict[str, Any], json.loads(text_out[start:end]))
            else:
                logger.warning("Failed to locate JSON brackets in LLM output.")
        except json.JSONDecodeError as e:
            logger.error(f"LLM output yielded invalid JSON: {e}. Raw text: {text_out}")
        except Exception as e:
            logger.exception(f"Unexpected error during semantic extraction: {e}")
        return {}

    def calculate_confidence(
        self, final: dict[str, Any], candidates: dict[str, list[Any]], has_isbn: bool
    ) -> ConfidenceScores:
        scores = {}
        for field in ["title", "author", "publisher", "published_date"]:
            if not final.get(field):
                scores[field] = 0.0
            else:
                vals = [v for v in candidates[field] if v]
                consistency = vals.count(final[field]) / len(vals) if vals else 0.5
                scores[field] = min(1.0, 0.5 + 0.5 * consistency)

        scores["isbn"] = 1.0 if has_isbn else 0.0
        return ConfidenceScores(**scores)
