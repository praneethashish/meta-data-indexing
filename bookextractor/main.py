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
async def extract(file: UploadFile = File(...)):
    filename = file.filename.lower()
    allowed_extensions = (".pdf", ".md", ".json")
    if not filename.endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_extensions}")
    
    # Save temp file
    import os
    temp_dir = "/tmp/bookextractor"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        p = get_pipeline()
        if filename.endswith(".pdf"):
            result = await p.process_pdf(temp_path)
        else:
            result = await p.process_text_file(temp_path)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@cli_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_file: Optional[str] = typer.Argument(None, help="Path to input file (.pdf, .md, .json)"),
    output_json: Optional[str] = typer.Argument(None, help="Path to output JSON"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Enable benchmark mode"),
    api: bool = typer.Option(False, "--api", help="Start FastAPI server"),
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
        filename = input_file.lower()
        if filename.endswith(".pdf"):
            result = await p.process_pdf(input_file, benchmark=benchmark)
        elif filename.endswith((".md", ".json")):
            result = await p.process_text_file(input_file, benchmark=benchmark)
        else:
            print(f"Error: Unsupported file type: {input_file}")
            raise typer.Exit(code=1)
        
        # Ensure output directory exists
        import os

        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_json}")

    asyncio.run(run_extraction())


if __name__ == "__main__":
    cli_app()
