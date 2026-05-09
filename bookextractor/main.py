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


def get_pipeline(max_model_len: int = 4096):
    global pipeline
    if pipeline is None:
        pipeline = ExtractionPipeline(max_model_len=max_model_len)
    return pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_api(host: str = "0.0.0.0", port: int = 8000) -> None:  # nosec B104
    print("Starting FastAPI server...")
    uvicorn.run(app, host=host, port=port)  # nosec


async def _extract_file(
    input_file: str,
    output_json: str,
    benchmark: bool = False,
    lang: OCRLanguage = OCRLanguage.ENGLISH,
    max_model_len: int = 4096,
) -> None:
    p = get_pipeline(max_model_len=max_model_len)
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
    max_model_len: int = typer.Option(  # noqa: B008
        4096,
        "--max-model-len",
        help="Maximum context length (reduce to save VRAM)",
    ),
) -> None:
    asyncio.run(_extract_file(input_file, output_json, benchmark=benchmark, lang=lang, max_model_len=max_model_len))


@cli_app.command("api")
def api_command(
    host: str = typer.Option("0.0.0.0", "--host", help="Host interface to bind the API server"),  # nosec B104
    port: int = typer.Option(8000, "--port", help="Port to bind the API server"),  # noqa: B008
) -> None:
    _run_api(host=host, port=port)


@cli_app.command("hardware-info")
def hardware_info_command() -> None:
    """Display detected hardware and suggested vLLM configuration."""
    from .hardware import get_vllm_config

    config = get_vllm_config()
    hw = config["detected_hardware"]

    print("\n--- Hardware Detection ---")
    print(f"Device:   {hw['device'].upper()}")
    print(f"Name:     {hw['name']}")
    print(f"Count:    {hw['count']}")
    print(f"Memory:   {hw['memory_gb']:.2f} GB")
    print(f"Precisions Supported: {', '.join(hw['precision_supported'])}")

    print("\n--- Optimized vLLM Configuration ---")
    print(f"Dtype:                   {config['dtype']}")
    print(f"GPU Memory Utilization:  {config['gpu_memory_utilization']}")
    print(f"Tensor Parallel Size:    {config['tensor_parallel_size']}")
    print(f"Target Device Override:  {os.getenv('VLLM_DEVICE', 'Not set (auto)')}")
    print("--------------------------\n")


model_app = typer.Typer(help="Manage Hugging Face models")
cli_app.add_typer(model_app, name="model", help="Download, list, and remove models")


@model_app.command("list")
def model_list_command() -> None:
    """List available models and their cache status."""
    from .hardware import get_vllm_config
    from .models_registry import AVAILABLE_MODELS, get_cached_model_size, is_model_cached

    try:
        config = get_vllm_config()
        vram = config["detected_hardware"]["memory_gb"]
    except Exception:
        vram = 0

    print(f"\n{'Model':<50} {'Type':<8} {'VRAM':<8} {'Cached':<8} {'Size':<10}")
    print("-" * 84)
    for m in AVAILABLE_MODELS:
        cached = is_model_cached(m["id"])
        size = get_cached_model_size(m["id"]) if cached else 0
        size_str = f"{size / 1024**3:.1f} GB" if size else "-"
        fits = "✅" if vram and m["min_vram_gb"] <= vram * 0.85 else ("⚠️" if vram else "?")
        print(
            f"{m['id']:<50} {m['type']:<8} {fits} {m['min_vram_gb']}GB{'':<3} "
            f"{'✓' if cached else '✗':<8} {size_str:<10}"
        )
    print()


@model_app.command("download")
def model_download_command() -> None:
    """Interactively download models from Hugging Face."""
    from huggingface_hub import snapshot_download
    from questionary import Style, checkbox

    from .models_registry import AVAILABLE_MODELS, is_model_cached

    choices = []
    for m in AVAILABLE_MODELS:
        cached = is_model_cached(m["id"])
        label = f"{'✓ ' if cached else '  '}{m['id']} — {m['desc']}"
        disabled = None
        choices.append({"name": label, "value": m["id"], "disabled": disabled})

    style = Style([("qmark", "fg:cyan bold"), ("question", "bold")])
    selected = checkbox("Select models to download:", choices=choices, style=style).ask()

    if not selected:
        print("No models selected.")
        return

    for model_id in selected:
        if is_model_cached(model_id):
            print(f"  ✓ {model_id} already cached, skipping.")
            continue
        print(f"\n  Downloading {model_id}...")
        try:
            snapshot_download(model_id, resume_download=True)
            print(f"  ✓ {model_id} downloaded successfully.")
        except Exception as e:
            print(f"  ✗ Failed to download {model_id}: {e}")

    print("\nDone.")


@model_app.command("remove")
def model_remove_command(
    query: str = typer.Argument(..., help="Model ID or name pattern to remove"),
) -> None:
    """Remove cached models by ID or fuzzy name."""
    from .models_registry import find_model_by_query, get_cached_models, remove_model_from_cache

    matches = find_model_by_query(query)
    if not matches:
        print(f"No models match '{query}'.")
        print("Cached models:", ", ".join(get_cached_models()))
        raise typer.Exit(code=1)

    if len(matches) == 1:
        model_id = matches[0]["id"]
    else:
        from questionary import select

        chosen = select(
            "Multiple matches. Choose one:",
            choices=[m["id"] for m in matches],
        ).ask()
        if not chosen:
            return
        model_id = chosen

    if remove_model_from_cache(model_id):
        print(f"  ✓ Removed {model_id}")
    else:
        print(f"  Not cached: {model_id}")


@model_app.command("cache")
def model_cache_command() -> None:
    """Show cache status and disk usage."""
    from .models_registry import get_cached_model_size, get_cached_models

    cached_ids = get_cached_models()
    if not cached_ids:
        print("\n  No models cached.\n")
        return

    total = 0
    print(f"\n{'Model':<50} {'Size':<12}")
    print("-" * 62)
    for model_id in cached_ids:
        size = get_cached_model_size(model_id)
        total += size
        size_str = f"{size / 1024**3:.2f} GB"
        print(f"{model_id:<50} {size_str:<12}")

    print("-" * 62)
    print(f"{'Total':<50} {total / 1024**3:.2f} GB")
    print()


if __name__ == "__main__":
    cli_app()
