import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from llama_cpp import Llama

from .external_api import lookup_isbn
from .image_utils import extract_image_metadata
from .models import BenchmarkResult, BookMetadata, ConfidenceScores, ExtractionResult, ImageMetadata
from .validation import extract_isbn_candidates
from .vparse_client import parse_pdf_via_vparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = Path(os.getenv("BOOKEXTRACTOR_MODELS_DIR", PROJECT_ROOT / "models"))

DEFAULT_VLM_FILENAME = "gemma-4-E4B-it-Q4_K_M.gguf"
DEFAULT_MMPROJ_FILENAME = "mmproj.gguf"

DEFAULT_VLM_URL = os.getenv(
    "VLM_MODEL_URL",
    "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
)
DEFAULT_MMPROJ_URL = os.getenv("MMPROJ_MODEL_URL")


def _download_file_if_missing(destination: Path, url: str, label: str) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "bookextractor/0.1.0"})

    print(f"{label} not found at {destination}. Downloading from {url} ...")
    try:
        with urlopen(request, timeout=60) as response, partial_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)

        if partial_path.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty")

        partial_path.replace(destination)
        print(f"Saved {label} to {destination}")
    except Exception as exc:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Failed to download {label} from {url}: {exc}") from exc


def resolve_model_paths() -> tuple[str, str | None]:
    models_dir = DEFAULT_MODELS_DIR

    configured_model_path = os.getenv("VLM_MODEL_PATH") or os.getenv("GEMMA_MODEL_PATH")
    model_path = Path(configured_model_path) if configured_model_path else models_dir / DEFAULT_VLM_FILENAME
    _download_file_if_missing(model_path, DEFAULT_VLM_URL, "GGUF model")

    configured_mmproj_path = os.getenv("MMPROJ_MODEL_PATH")
    mmproj_path: Path | None = None
    if configured_mmproj_path:
        mmproj_path = Path(configured_mmproj_path)
        if not mmproj_path.exists():
            raise ValueError(f"Configured MMPROJ_MODEL_PATH does not exist: {mmproj_path}")
    elif DEFAULT_MMPROJ_URL:
        mmproj_path = models_dir / DEFAULT_MMPROJ_FILENAME
        _download_file_if_missing(mmproj_path, DEFAULT_MMPROJ_URL, "Projection model")

    resolved_model_path = str(model_path)
    resolved_mmproj_path = str(mmproj_path) if mmproj_path else None

    # Keep environment values in sync so downstream calls use resolved local paths.
    os.environ["VLM_MODEL_PATH"] = resolved_model_path
    os.environ["GEMMA_MODEL_PATH"] = resolved_model_path
    if resolved_mmproj_path:
        os.environ["MMPROJ_MODEL_PATH"] = resolved_mmproj_path

    return resolved_model_path, resolved_mmproj_path


class ExtractionPipeline:
    def __init__(self, model_path: str | None = None):
        resolved_model_path, _ = resolve_model_paths()
        effective_model_path = model_path or resolved_model_path

        if not os.path.exists(effective_model_path):
            raise ValueError(f"Model path does not exist after resolution: {effective_model_path}")

        self.llm = Llama(model_path=effective_model_path, n_ctx=2048, verbose=False)

    async def process_pdf(self, pdf_path: str, benchmark: bool = False) -> Dict[str, Any]:
        # Call vParse OCR API
        vparse_response = await parse_pdf_via_vparse(pdf_path)
        
        # Extract text from vParse response
        filename = os.path.basename(pdf_path).rsplit(".", 1)[0]
        result_data = vparse_response.get("results", {}).get(filename, {})
        
        full_text = result_data.get("md_content", "")
        if not full_text:
            content_list = result_data.get("content_list", [])
            if isinstance(content_list, list):
                full_text = "\n".join([item.get("text", "") for item in content_list if isinstance(item, dict)])
            elif isinstance(content_list, str):
                full_text = content_list

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
        prompt = f"""<|system|>
You are an expert multilingual metadata extractor.
Fields to extract: title, author, publisher, published_date
Rules:
- Return STRICT JSON only.
- Do NOT guess.
- If uncertain, return null.
- Do NOT include ISBN.
- Text may be noisy OCR output in English, Telugu, or Hindi. Look for names and titles carefully.
<|user|>
Text:
{text[:1500]}
<|assistant|>
"""
        try:
            output = self.llm(
                prompt,
                max_tokens=256,
                stop=["<|end|>", "\n\n"],
                echo=False,
                stream=False,
            )
            if isinstance(output, Iterator):
                return {}

            text_out = output["choices"][0]["text"].strip()
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
