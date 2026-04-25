import os
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from collections.abc import Iterator
import fitz
from PIL import Image
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from .models import BookMetadata, ConfidenceScores, BenchmarkResult
from .pdf_utils import extract_keyword_bboxes, get_page_image
from .image_utils import crop_regions, crop_bbox, combine_regions
from .ocr import OCRScanner
from .external_api import lookup_isbn

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


def resolve_model_paths() -> tuple[str, Optional[str]]:
    models_dir = DEFAULT_MODELS_DIR

    configured_model_path = os.getenv("VLM_MODEL_PATH") or os.getenv("GEMMA_MODEL_PATH")
    model_path = Path(configured_model_path) if configured_model_path else models_dir / DEFAULT_VLM_FILENAME
    _download_file_if_missing(model_path, DEFAULT_VLM_URL, "GGUF model")

    configured_mmproj_path = os.getenv("MMPROJ_MODEL_PATH")
    mmproj_path: Optional[Path] = None
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
    def __init__(self, model_path: Optional[str] = None):
        self.ocr = OCRScanner()
        resolved_model_path, _ = resolve_model_paths()
        effective_model_path = model_path or resolved_model_path

        if not os.path.exists(effective_model_path):
            raise ValueError(f"Model path does not exist after resolution: {effective_model_path}")

        self.llm = Llama(model_path=effective_model_path, n_ctx=2048, verbose=False)

    async def process_pdf(self, pdf_path: str, benchmark: bool = False) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)
        # Select ONLY first 7 pages as requested
        page_indices = list(range(min(7, len(doc))))
        
        all_candidates = {
            "title": [], "author": [], "publisher": [], "published_date": [], "isbn": []
        }
        
        debug_info = {
            "pages_used": page_indices,
            "isbn_candidates": [],
            "ocr_text_snippets": [],
            "vl_raw_outputs": []
        }

        for idx in page_indices:
            page = doc[idx]
            pix = get_page_image(page)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            
            # Adaptive Region Extraction
            kw_bboxes = extract_keyword_bboxes(page)
            kw_crops = [crop_bbox(img, rect, page.rect) for rect in kw_bboxes]
            std_crops = crop_regions(img)
            all_crops = combine_regions(std_crops, kw_crops)
            
            # OCR for ISBN and general text
            isbns = self.ocr.find_isbns(all_crops)
            all_candidates["isbn"].extend(isbns)
            debug_info["isbn_candidates"].extend(isbns)
            
            # Extract full text for LLM from standard crops or full page
            full_text = self.ocr.scan_image(img)
            debug_info["ocr_text_snippets"].append(full_text[:500])
            
            # LLM Semantic Extraction
            llm_result = self.extract_semantic_fields(full_text)
            debug_info["vl_raw_outputs"].append(llm_result)
            
            for field in ["title", "author", "publisher", "published_date"]:
                if llm_result.get(field):
                    all_candidates[field].append(llm_result[field])

        # Field-wise Merge
        final_data = self.merge_candidates(all_candidates)
        
        # ISBN Validation & External lookup
        isbn = None
        if all_candidates["isbn"]:
            # Pick first valid ISBN
            isbn = all_candidates["isbn"][0]
            ext_data = await lookup_isbn(isbn)
            for k, v in ext_data.items():
                if v: # Override if external data is available
                    final_data[k] = v
        
        final_data["isbn"] = isbn
        
        # Confidence Scoring
        confidence = self.calculate_confidence(final_data, all_candidates, bool(isbn))
        
        result = BookMetadata(
            title=final_data.get("title"),
            author=final_data.get("author"),
            publisher=final_data.get("publisher"),
            isbn=final_data.get("isbn"),
            published_date=final_data.get("published_date"),
            confidence=confidence
        )

        if benchmark:
            return BenchmarkResult(result=result, debug=debug_info).dict()
        return result.dict()

    def extract_semantic_fields(self, text: str) -> Dict[str, Any]:
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
            # Try to find JSON in output
            start = text_out.find('{')
            end = text_out.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(text_out[start:end])
        except Exception:
            pass
        return {}

    def merge_candidates(self, candidates: Dict[str, List[str]]) -> Dict[str, Any]:
        merged = {}
        for field in ["title", "author", "publisher", "published_date"]:
            vals = [v for v in candidates[field] if v]
            if not vals:
                merged[field] = None
                continue
            # Prefer longest string for completeness
            merged[field] = max(vals, key=len)
        return merged

    def calculate_confidence(self, final: Dict[str, Any], candidates: Dict[str, List[str]], has_isbn: bool) -> ConfidenceScores:
        scores = {}
        for field in ["title", "author", "publisher", "published_date"]:
            if not final.get(field):
                scores[field] = 0.0
            else:
                # If we have multiple consistent candidates, higher confidence
                vals = [v for v in candidates[field] if v]
                consistency = vals.count(final[field]) / len(vals) if vals else 0.5
                scores[field] = min(1.0, 0.5 + 0.5 * consistency)
        
        scores["isbn"] = 1.0 if has_isbn else 0.0
        return ConfidenceScores(**scores)
