"""Download VParse OCR models into /models volume.

Usage:
    python setup_models.py

Note: Gemma-4 LLM is no longer downloaded here.
vLLM downloads models directly from HuggingFace using HF_TOKEN.
The HuggingFace cache is stored at ~/.cache/huggingface/hub/ (host bind mount).

For Gemma-4 via vLLM, ensure HF_TOKEN is set and accept model terms at:
https://huggingface.co/google/gemma-4-E4B-it
"""

from pathlib import Path

MODELS_DIR = Path("/models")

PIPELINE_HF_REPO = "opendatalab/PDF-Extract-Kit-1.0"

PIPELINE_PATTERNS = [
    "models/Layout/YOLO/*",
    "models/MFD/YOLO/*",
    "models/MFR/unimernet_hf_small_2503/*",
    "models/MFR/pp_formulanet_plus_m/*",
    "models/OCR/paddleocr_torch/*",
    "models/ReadingOrder/layout_reader/*",
    "models/TabRec/*",
    "models/TabCls/*",
    "models/OriCls/*",
    "models/README.md",
]


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
        print("  ✓ Gemma model ready.")
    except Exception as e:
        print(f"  ✗ Error downloading Gemma: {e}")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    download_pipeline_models()

    print("\n══════════════════════════")
    print("   ALL MODELS READY")
    print("══════════════════════════")
    print("\nNote: Gemma-4 LLM is downloaded by vLLM from HuggingFace.")
    print("Ensure HF_TOKEN is set and ~/.cache/huggingface/hub is mounted.\n")
