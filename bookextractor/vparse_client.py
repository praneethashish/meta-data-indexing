import os
import httpx
from typing import Dict, Any, Optional

VPARSE_API_URL = os.getenv("VPARSE_API_URL", "http://localhost:8000/file_parse")

async def parse_pdf_via_vparse(file_path: str) -> Dict[str, Any]:
    """
    Sends a PDF to the mineru-dots vParse API for OCR processing.
    Returns the parsed JSON response.
    """
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            files = {"files": (os.path.basename(file_path), f, "application/pdf")}
            data = {"backend": "pipeline", "return_md": "true", "return_content_list": "true"}
            try:
                response = await client.post(VPARSE_API_URL, files=files, data=data, timeout=300.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                raise RuntimeError(f"vParse API call failed: {e}") from e
