import json
import os
import re
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

from .external_api import lookup_isbn
from .image_utils import extract_image_metadata
from .models import BenchmarkResult, BookMetadata, ConfidenceScores, ExtractionResult, ImageMetadata
from .validation import extract_isbn_candidates
from .vparse_client import parse_pdf_via_vparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = Path(os.getenv("BOOKEXTRACTOR_MODELS_DIR", PROJECT_ROOT / "models"))


class ExtractionPipeline:
    def __init__(self, model_id: str | None = None):
        model = model_id or os.getenv("VLLM_MODEL", "google/gemma-4-E4B-it")
        tensor_parallel_size = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=8192,
        )

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
                try:
                    content_list = json.loads(content_list)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(content_list, list):
                full_text = "\n".join(
                    item.get("text", "") for item in content_list
                    if isinstance(item, dict) and item.get("text")
                )

        # Fallback to md_content
        if not full_text and isinstance(result_data, dict):
            full_text = result_data.get("md_content", "")

        # Clean up the text for LLM
        full_text = re.sub(r'!\[.*?\]\(.*?\)\s*', '', full_text)  # Remove image refs
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Collapse whitespace
        full_text = full_text.strip()

        return await self.extract_from_text(full_text, benchmark=benchmark)

    async def process_image(self, image_path: str, benchmark: bool = False) -> dict[str, Any]:
        metadata_dict = extract_image_metadata(image_path)
        img_meta = ImageMetadata(**metadata_dict)
        result = ExtractionResult(image_metadata=img_meta)

        if benchmark:
            return BenchmarkResult(result=result).dict()
        return result.dict()

    async def process_text_file(self, file_path: str, benchmark: bool = False) -> dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return await self.extract_from_text(content, benchmark=benchmark)

    async def extract_from_text(self, text: str, benchmark: bool = False) -> dict[str, Any]:
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
            return BenchmarkResult(result=result, debug=debug_info).dict()
        return result.dict()

    def extract_semantic_fields(self, text: str) -> dict[str, Any]:
        prompt = f"""You are an expert book metadata extractor. Your task is to identify and extract structured metadata from noisy OCR text of scanned book pages.

The text below was extracted via OCR from scanned book pages and may contain:
- OCR errors, garbled characters, or misread words
- Mixed languages (English, Telugu, Hindi)
- Publishing information like edition details, print runs, and pricing
- Copyright notices with author names
- Publisher addresses and contact information

Extract the following fields and return ONLY a valid JSON object:
- "title": The book's title (look for prominent text, large headings, or text on the title page)
- "author": The author's full name (look near "By", "©", "Written by", or Telugu/Hindi equivalents)
- "publisher": The publisher's name (look near "Published by", "ప్రచురణ", "प्रकाशक", or publishing house names)
- "published_date": The earliest publication date (look for years like 1996, 2004 near "First Edition", "ముద్రణ", "संस्करण")

Rules:
- Return STRICT JSON only — no explanation, no markdown, no extra text.
- If a field cannot be confidently determined, set its value to null.
- Do NOT fabricate or guess values. Only extract what is clearly present.
- Do NOT include ISBN (it is extracted separately).
- Prefer the original/first edition date over reprint dates.

OCR Text:
{text[:3000]}

JSON:
"""
        try:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=256, stop=["```"])
            outputs = self.llm.generate([prompt], sampling_params)
            text_out = outputs[0].outputs[0].text.strip()
            start = text_out.find("{")
            end = text_out.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(text_out[start:end])
        except Exception:
            pass
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
