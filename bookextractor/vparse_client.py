import os
from typing import Any

import httpx

VPARSE_API_URL = os.getenv("VPARSE_API_URL", "http://localhost:8000/file_parse")


async def parse_pdf_via_vparse(file_path: str, lang: str = "en") -> dict[str, Any]:
    """
    Sends a PDF to the mineru-dots vParse API for OCR processing.

    Args:
        file_path: Path to the PDF file.
        lang: PaddleOCR language pack to use.
              Supported: "en" (English), "te" (Telugu+English),
              "devanagari" (Hindi+English), "ch" (Chinese+English).

    Returns the parsed JSON response.
    """
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            files = {"files": (os.path.basename(file_path), f, "application/pdf")}
            data = {
                "backend": "pipeline",
                "return_md": "true",
                "return_content_list": "true",
                "lang_list": lang,
            }
            try:
                response = await client.post(VPARSE_API_URL, files=files, data=data, timeout=900.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                raise RuntimeError(f"vParse API call failed: {e}") from e
