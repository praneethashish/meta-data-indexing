import asyncio
import json
import os

import typer
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from .pipeline import ExtractionPipeline

app = FastAPI()
cli_app = typer.Typer()
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ExtractionPipeline()
    return pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):  # noqa: B008
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    filename = file.filename.lower()
    allowed_extensions = (".pdf", ".md", ".json", ".jpg", ".jpeg", ".png", ".webp", ".tiff")
    if not filename.endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_extensions}")

    # Save temp file
    temp_dir = "/tmp/bookextractor"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        p = get_pipeline()
        if filename.endswith(".pdf"):
            result = await p.process_pdf(temp_path)
        elif filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff")):
            result = await p.process_image(temp_path)
        else:
            result = await p.process_text_file(temp_path)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@cli_app.callback(invoke_without_command=True)
def main(
    _ctx: typer.Context,
    input_file: str | None = typer.Argument(None, help="Input file path (.pdf, .md, .json, .jpg, .png, .webp, .tiff)"),  # noqa: B008
    output_json: str | None = typer.Argument(None, help="Path to output JSON"),  # noqa: B008
    benchmark: bool = typer.Option(False, "--benchmark", help="Enable benchmark mode"),  # noqa: B008
    api: bool = typer.Option(False, "--api", help="Start FastAPI server"),  # noqa: B008
):
    if api:
        print("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    if not input_file or not output_json:
        print("Error: Missing arguments. Usage: bookextractor <input_file> <output.json> or bookextractor --api")
        raise typer.Exit(code=1)

    async def run_extraction():
        p = get_pipeline()
        filename = input_file.lower() if input_file else ""

        if filename.endswith(".pdf"):
            result = await p.process_pdf(input_file, benchmark=benchmark)
        elif filename.endswith((".md", ".json")):
            result = await p.process_text_file(input_file, benchmark=benchmark)
        elif filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff")):
            result = await p.process_image(input_file, benchmark=benchmark)
        else:
            print(f"Error: Unsupported file type: {input_file}")
            raise typer.Exit(code=1)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_json}")

    asyncio.run(run_extraction())


if __name__ == "__main__":
    cli_app()
