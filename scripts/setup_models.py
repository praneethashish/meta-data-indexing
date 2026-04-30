"""Download only the models required for the selected mode into /models volume.

Usage:
    python setup_models.py              # downloads pipeline (default) + gemma
    python setup_models.py --mode vlm   # downloads vlm + gemma
"""
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

# ── Configuration ────────────────────────────────────────────────────
MODELS_DIR = Path("/models")

GEMMA_URL = "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf"
GEMMA_FILENAME = "gemma-4-E4B-it-Q4_K_M.gguf"

PIPELINE_HF_REPO = "opendatalab/PDF-Extract-Kit-1.0"

# Only the model folders actually used by pipeline/paddle mode.
# Reference: mineru-dots/vparse/utils/enum_class.py  ModelPath
PIPELINE_PATTERNS = [
    "models/Layout/YOLO/*",                    # doclayout_yolo  (~55 MB)
    "models/MFD/YOLO/*",                       # yolo_v8_mfd     (~50 MB)
    "models/MFR/unimernet_hf_small_2503/*",    # formula recog   (~500 MB)
    "models/MFR/pp_formulanet_plus_m/*",       # formula recog   (~200 MB)
    "models/OCR/paddleocr_torch/*",            # paddle OCR      (~200 MB)
    "models/ReadingOrder/layout_reader/*",     # reading order   (~500 MB)
    "models/TabRec/*",                         # table recog     (~100 MB)
    "models/TabCls/*",                         # table classify  (~20 MB)
    "models/OriCls/*",                         # orientation cls (~20 MB)
    "models/README.md",
]


def download_file(url: str, destination: Path) -> None:
    """Download a single file via HTTP, skipping if it already exists."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  ✓ {destination.name} already exists. Skipping.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  ↓ Downloading {destination.name} ...")
    request = Request(url, headers={"User-Agent": "model-downloader/1.0"})
    try:
        with urlopen(request) as resp, partial.open("wb") as out:
            shutil.copyfileobj(resp, out)
        partial.replace(destination)
        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"  ✓ {destination.name} ({size_mb:.0f} MB)")
    except Exception as e:
        if partial.exists():
            partial.unlink()
        print(f"  ✗ Error downloading {destination.name}: {e}")


def download_pipeline_models() -> None:
    """Download only the pipeline/paddle OCR models from HuggingFace."""
    print("\n── Downloading Pipeline/Paddle OCR models ──")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=PIPELINE_HF_REPO,
            local_dir=MODELS_DIR / "opendatalab" / "PDF-Extract-Kit-1.0",
            allow_patterns=PIPELINE_PATTERNS,
        )
        print("  ✓ Pipeline models ready.")
    except Exception as e:
        print(f"  ✗ Error downloading pipeline models: {e}")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gemma LLM for metadata extraction
    print("\n── Downloading Gemma LLM ──")
    download_file(GEMMA_URL, MODELS_DIR / GEMMA_FILENAME)

    # 2. OCR models for VParse pipeline mode
    download_pipeline_models()

    print("\n══════════════════════════")
    print("   ALL MODELS READY")
    print("══════════════════════════\n")
