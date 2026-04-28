import asyncio
import json
import os
from typing import Annotated

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
async def extract(file: Annotated[UploadFile, File()]):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Save temp file
    temp_path = f"/tmp/{os.path.basename(file.filename)}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        p = get_pipeline()
        result = await p.process_pdf(temp_path)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@cli_app.callback(invoke_without_command=True)
def main(
    _ctx: typer.Context,
    input_pdf: str | None = typer.Argument(None, help="Path to input PDF"),
    output_json: str | None = typer.Argument(None, help="Path to output JSON"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Enable benchmark mode"),
    api: bool = typer.Option(False, "--api", help="Start FastAPI server"),
):
    if api:
        print("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    if not input_pdf or not output_json:
        print("Error: Missing arguments. Usage: bookextractor <input.pdf> <output.json> or bookextractor --api")
        raise typer.Exit(code=1)

    async def run_extraction():
        p = get_pipeline()
        result = await p.process_pdf(input_pdf, benchmark=benchmark)

        # Ensure output directory exists
        import os

        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_json}")

    asyncio.run(run_extraction())


if __name__ == "__main__":
    cli_app()
