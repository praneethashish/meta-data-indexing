import os
from typing import Any, cast

import httpx

from .config import settings


async def parse_pdf_via_vparse(file_path: str, lang: str = "en") -> dict[str, Any]:
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
                response = await client.post(settings.VPARSE_API_URL, files=files, data=data, timeout=settings.VPARSE_TIMEOUT)
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
            except Exception as e:
                raise RuntimeError(f"vParse API call failed: {e}") from e
