"""Download required runtime models into the shared Docker volume.

Usage:
    python setup_models.py
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_MODEL_PATH = "/models/vllm/google/gemma-4-E4B-it"


def _path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def download_llm() -> None:
    model_id = os.getenv("VLLM_MODEL_ID", DEFAULT_MODEL_ID)
    target_path = Path(os.getenv("VLLM_MODEL_PATH", DEFAULT_MODEL_PATH))
    hf_home = Path(os.getenv("HF_HOME", "/models/huggingface"))

    target_path.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    if _path_has_files(target_path):
        print(f"\n── LLM already present at {target_path}; skipping download ──")
        return

    print(f"\n── Downloading LLM ({model_id}) to {target_path} ──")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_path),
        )
        print(f"  ✓ LLM model {model_id} ready at {target_path}.")
    except Exception as e:
        raise RuntimeError(f"Failed to download LLM model {model_id}: {e}") from e


if __name__ == "__main__":
    download_llm()

    print("\n══════════════════════════")
    print("   ALL PREFETCH MODELS READY")
    print("══════════════════════════")
    print("\nNote: Models are stored in the shared Docker volume and reused across container restarts.\n")
