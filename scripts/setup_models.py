import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── Configuration ────────────────────────────────────────────────────
MODELS_DIR = Path("/models")

GEMMA_REPO = "unsloth/gemma-4-E4B-it-GGUF"
GEMMA_FILENAME = "gemma-4-E4B-it-Q4_K_M.gguf"

def download_gemma() -> None:
    print(f"\n── Downloading Gemma LLM ({GEMMA_FILENAME}) ──")
    try:
        hf_hub_download(
            repo_id=GEMMA_REPO,
            filename=GEMMA_FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print("  ✓ Gemma model ready.")
    except Exception as e:
        print(f"  ✗ Error downloading Gemma: {e}")

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    download_gemma()

    print("\n══════════════════════════")
    print("   ALL MODELS READY")
    print("══════════════════════════\n")

