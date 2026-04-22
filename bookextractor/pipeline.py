import json
import fitz
from PIL import Image
from typing import List, Dict, Any
from llama_cpp import Llama
from .models import BookMetadata, ConfidenceScores, BenchmarkResult
from .pdf_utils import extract_keyword_bboxes, get_page_image
from .image_utils import crop_regions, crop_bbox, combine_regions
from .ocr import OCRScanner
from .external_api import lookup_isbn

DEFAULT_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", "/home/praneethashish/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/ce152932ac27bc40bc9c727386760424d50bb456/gemma-4-E4B-it-Q4_K_M.gguf")

class ExtractionPipeline:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.ocr = OCRScanner()
        self.llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

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
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
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
            output = self.llm(prompt, max_tokens=256, stop=["<|end|>", "\n\n"], echo=False)
            text_out = output['choices'][0]['text'].strip()
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
