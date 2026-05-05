import asyncio
import json
import os
from enum import Enum

import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .pipeline import ExtractionPipeline


class OCRLanguage(str, Enum):
    """Supported PaddleOCR language packs."""

    ENGLISH = "en"
    TELUGU = "te"
    HINDI = "devanagari"


app = FastAPI()
cli_app = typer.Typer()
pipeline = None
ALLOWED_EXTENSIONS = (".pdf", ".md", ".json", ".jpg", ".jpeg", ".png", ".webp", ".tiff")


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ExtractionPipeline()
    return pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_api(host: str = "0.0.0.0", port: int = 8000) -> None:  # nosec B104
    print("Starting FastAPI server...")
    uvicorn.run(app, host=host, port=port)  # nosec


async def _extract_file(
    input_file: str, output_json: str, benchmark: bool = False, lang: OCRLanguage = OCRLanguage.ENGLISH
) -> None:
    p = get_pipeline()
    filename = input_file.lower()

    if filename.endswith(".pdf"):
        result = await p.process_pdf(input_file, benchmark=benchmark, lang=lang.value)
    elif filename.endswith((".md", ".json")):
        result = await p.process_text_file(input_file, benchmark=benchmark)
    elif filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff")):
        result = await p.process_image(input_file, benchmark=benchmark)
    else:
        print(f"Error: Unsupported file type: {input_file}")
        raise typer.Exit(code=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_json}")


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),  # noqa: B008
    lang: OCRLanguage = Form(  # noqa: B008
        OCRLanguage.ENGLISH,
        description="OCR language pack: 'en' (English), 'te' (Telugu+English), 'devanagari' (Hindi+English)",
    ),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    filename = file.filename.lower()
    if not filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")

    # Save temp file
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        try:
            p = get_pipeline()
            if filename.endswith(".pdf"):
                result = await p.process_pdf(temp_path, lang=lang.value)
            elif filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff")):
                result = await p.process_image(temp_path)
            else:
                result = await p.process_text_file(temp_path)
            return result
        finally:
            pass  # TemporaryDirectory handles cleanup


@cli_app.callback()
def main(
    ctx: typer.Context,
    api: bool = typer.Option(False, "--api", help="Start FastAPI server"),  # noqa: B008
):
    if ctx.invoked_subcommand is not None:
        return

    if api:
        _run_api()
        return

    print(
        "Error: Missing arguments. Usage: bookextractor extract <input_file> <output.json> [--lang te], "
        "or bookextractor api"
    )
    raise typer.Exit(code=1)


@cli_app.command("extract")
def extract_command(
    input_file: str = typer.Argument(..., help="Input file path (.pdf, .md, .json, .jpg, .png, .webp, .tiff)"),  # noqa: B008
    output_json: str = typer.Argument(..., help="Path to output JSON"),  # noqa: B008
    benchmark: bool = typer.Option(False, "--benchmark", help="Enable benchmark mode"),  # noqa: B008
    lang: OCRLanguage = typer.Option(  # noqa: B008
        OCRLanguage.ENGLISH,
        "--lang",
        help="OCR language pack for PDF extraction: en, te, devanagari",
    ),
) -> None:
    asyncio.run(_extract_file(input_file, output_json, benchmark=benchmark, lang=lang))


@cli_app.command("api")
def api_command(
    host: str = typer.Option("0.0.0.0", "--host", help="Host interface to bind the API server"),  # nosec B104
    port: int = typer.Option(8000, "--port", help="Port to bind the API server"),  # noqa: B008
) -> None:
    _run_api(host=host, port=port)


if __name__ == "__main__":
    cli_app()
