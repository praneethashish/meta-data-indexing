import asyncio
import json
import logging
import os
import pathlib
import uuid
from enum import Enum
from functools import lru_cache

import typer
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from .config import settings
from .pipeline import ExtractionPipeline
from .tasks import celery_app, extract_image_task, extract_pdf_task, extract_text_task

logger = logging.getLogger(__name__)


class OCRLanguage(str, Enum):
    """Supported PaddleOCR language packs."""

    ENGLISH = "en"
    TELUGU = "te"
    HINDI = "devanagari"


app = FastAPI()
cli_app = typer.Typer()


@lru_cache
def get_vision_pipeline():
    return ExtractionPipeline(load_vlm=False)


@app.get("/health")
def health():
    from .model_manager import ModelManager

    manager = ModelManager.get_instance()
    return {"status": "ok", "model_ready": manager.is_loaded()}


@app.post("/extract/async")
async def extract_async(
    file: UploadFile = File(...),  # noqa: B008
    lang: OCRLanguage = Form(  # noqa: B008
        OCRLanguage.ENGLISH,
        description="OCR language pack: 'en' (English), 'te' (Telugu+English), 'devanagari' (Hindi+English)",
    ),
    use_vlm: bool = Form(False, description="Run deep Vision analysis on images (requires GPU)"),  # noqa: B008
):
    """Submit an extraction job to the background queue."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    filename = file.filename.lower()
    if not filename.endswith(settings.ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {settings.ALLOWED_EXTENSIONS}")

    # Save to persistent upload dir for worker access
    file_id = str(uuid.uuid4())
    safe_extension = pathlib.Path(file.filename).suffix.lower()
    save_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{safe_extension}")

    with open(save_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        if filename.endswith(".pdf"):
            task = extract_pdf_task.delay(save_path, lang=lang.value)
        elif filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif")):
            target_queue = "vlm_queue" if use_vlm else "default_queue"
            task = extract_image_task.apply_async(
                args=[save_path],
                kwargs={"use_vlm": use_vlm},
                queue=target_queue,
            )
        else:
            task = extract_text_task.delay(save_path, lang=lang.value)

        return {"job_id": task.id, "status": "submitted"}
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}") from e


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status or result of a background extraction job."""
    task = celery_app.AsyncResult(job_id)
    response = {
        "job_id": job_id,
        "status": task.state,
        "ready": task.ready(),
    }

    if task.ready():
        if task.successful():
            response["result"] = task.result
        else:
            response["error"] = str(task.result)

    return response


def _run_api(host: str = "", port: int = 0) -> None:
    host = host or settings.API_HOST
    port = port or settings.API_PORT
    logger.info("Starting FastAPI server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


async def _extract_file(
    input_file: str,
    output_json: str,
    benchmark: bool = False,
    lang: OCRLanguage = OCRLanguage.ENGLISH,
    max_model_len: int = 4096,
    use_vlm: bool = False,
) -> None:
    filename = input_file.lower()
    is_image = filename.endswith(settings.IMAGE_EXTENSIONS)
    p = ExtractionPipeline(max_model_len=max_model_len, load_vlm=(not is_image) or use_vlm)

    if filename.endswith(".pdf"):
        result = await p.process_pdf(input_file, benchmark=benchmark, lang=lang.value)
    elif filename.endswith((".md", ".json", ".txt")):
        result = await p.process_text_file(input_file, benchmark=benchmark, lang=lang.value)
    elif is_image:
        result = await p.process_image(input_file, benchmark=benchmark)
    else:
        print(f"Error: Unsupported file type: {input_file}")
        raise typer.Exit(code=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", output_json)


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),  # noqa: B008
    vision_pipeline: ExtractionPipeline = Depends(get_vision_pipeline),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    filename = file.filename.lower()
    if not filename.endswith(settings.IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Synchronous extraction is only supported for images. Use /extract/async for PDFs/Text.",
        )

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        result = await vision_pipeline.process_image(temp_path)
        return result


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
    input_file: str = typer.Argument(
        ..., help="Input file path (.pdf, .md, .json, .txt, .jpg, .png, .webp, .tiff, .tif)"
    ),  # noqa: B008, E501
    output_json: str = typer.Argument(..., help="Path to output JSON"),  # noqa: B008
    benchmark: bool = typer.Option(False, "--benchmark", help="Enable benchmark mode"),  # noqa: B008
    lang: OCRLanguage = typer.Option(  # noqa: B008
        OCRLanguage.ENGLISH,
        "--lang",
        help="OCR language pack for PDF extraction: en, te, devanagari",
    ),
    use_vlm: bool = typer.Option(False, "--vlm", help="Enable VLM analysis for images"),  # noqa: B008
    max_model_len: int = typer.Option(  # noqa: B008
        settings.MAX_MODEL_LEN,
        "--max-model-len",
        help="Maximum context length (reduce to save VRAM)",
    ),
) -> None:
    asyncio.run(
        _extract_file(
            input_file, output_json, benchmark=benchmark, lang=lang, max_model_len=max_model_len, use_vlm=use_vlm
        )
    )


@cli_app.command("api")
def api_command(
    host: str = typer.Option(settings.API_HOST, "--host", help="Host interface to bind the API server"),  # nosec B104
    port: int = typer.Option(settings.API_PORT, "--port", help="Port to bind the API server"),  # noqa: B008
) -> None:
    _run_api(host=host, port=port)


@cli_app.command("worker")
def worker_command(
    queue: str = typer.Option(settings.CELERY_DEFAULT_QUEUE, "--queue", "-q", help="Celery queue to listen to"),
    concurrency: int = typer.Option(settings.DEFAULT_WORKER_CONCURRENCY, "--concurrency", "-c", help="Number of concurrent worker processes"),  # noqa: E501
):
    """Start a Celery worker for background processing."""
    logger.info("Starting Celery worker for queue: %s (concurrency: %s)", queue, concurrency)
    import subprocess  # nosec

    cmd = [
        "celery",
        "-A",
        "bookextractor.tasks",
        "worker",
        "-Q",
        queue,
        "--concurrency",
        str(concurrency),
        "--loglevel=info",
    ]
    subprocess.run(cmd)  # nosec


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
            snapshot_download(model_id)
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
