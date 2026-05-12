"""Download required runtime models into the shared Docker volume.

Usage:
    python setup_models.py
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def download_llm() -> None:
    """Download the LLM model using huggingface_hub.

    The model is downloaded to the directory specified by HF_HOME.
    vLLM and other tools will automatically find it there via its model ID.
    """
    model_id = os.getenv("VLLM_MODEL_ID", DEFAULT_MODEL_ID)
    hf_cache = Path(os.getenv("HF_HOME", "/models/huggingface"))

    print(f"\n── Preparing LLM ({model_id}) ──")
    print(f"   Cache Directory: {hf_cache}")

    try:
        # snapshot_download handles skipping if already downloaded efficiently
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(hf_cache),
        )
        print(f"  ✓ LLM model {model_id} ready in cache: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download LLM model {model_id}: {e}") from e


if __name__ == "__main__":
    download_llm()

    print("\n══════════════════════════")
    print("   ALL PREFETCH MODELS READY")
    print("══════════════════════════")
    print("\nNote: Models are stored in the shared Hugging Face cache and reused across container restarts.\n")
