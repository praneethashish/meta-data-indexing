from pathlib import Path
from typing import Any

from .config import settings

_base_cache = settings.HF_HOME
HF_CACHE_DIR = Path(_base_cache) / "hub" if not _base_cache.endswith("hub") else Path(_base_cache)

AVAILABLE_MODELS: list[dict[str, Any]] = [
    {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "name": "Qwen 2.5 VL 7B",
        "type": "vision",
        "min_vram_gb": 16,
        "desc": "Full precision vision-language model for large GPUs",
    },
    {
        "id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "name": "Qwen3 VL 30B (MoE, 3B active)",
        "type": "vision",
        "min_vram_gb": 6,
        "desc": "MoE vision-language — only 3B active params, best T4 fit",
    },
    {
        "id": "Qwen/Qwen3.5-9B",
        "name": "Qwen 3.5 9B",
        "type": "text",
        "min_vram_gb": 10,
        "desc": "Text-only, good for metadata extraction on capable GPUs",
    },
    {
        "id": "google/gemma-3-27b-it",
        "name": "Gemma 3 27B",
        "type": "text",
        "min_vram_gb": 28,
        "desc": "High quality text, needs A100/H100",
    },
    {
        "id": "google/gemma-4-31b-it",
        "name": "Gemma 4 31B",
        "type": "text",
        "min_vram_gb": 32,
        "desc": "Latest Gemma, requires high-end GPU",
    },
    {
        "id": "google/gemma-4-E4B-it",
        "name": "Gemma 4 E4B",
        "type": "text",
        "min_vram_gb": 8,
        "desc": "Gemma 4 E4B — compact, excellent for metadata extraction",
    },
]


def cache_dir_for_model(model_id: str) -> Path:
    org, name = model_id.split("/", 1)
    return HF_CACHE_DIR / f"models--{org}--{name.replace('.', '__dot__')}"


def is_model_cached(model_id: str) -> bool:
    cache_path = cache_dir_for_model(model_id)
    if not cache_path.exists():
        return False
    blobs = list(cache_path.glob("snapshots/*/*.safetensors"))
    return len(blobs) > 0


def get_cached_model_size(model_id: str) -> int:
    cache_path = cache_dir_for_model(model_id)
    if not cache_path.exists():
        return 0
    total = 0
    for f in cache_path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def get_cached_models() -> list[str]:
    if not HF_CACHE_DIR.exists():
        return []
    cached = []
    for d in HF_CACHE_DIR.iterdir():
        if d.name.startswith("models--"):
            parts = d.name.replace("models--", "", 1).split("--", 1)
            if len(parts) == 2:
                org, name = parts
                name = name.replace("__dot__", ".")
                model_id = f"{org}/{name}"
                if is_model_cached(model_id):
                    cached.append(model_id)
    return cached


def remove_model_from_cache(model_id: str) -> bool:
    cache_path = cache_dir_for_model(model_id)
    if not cache_path.exists():
        return False
    import shutil

    shutil.rmtree(cache_path)
    return True


def find_model_by_query(query: str) -> list[dict[str, Any]]:
    query_lower = query.lower().replace("-", "").replace("_", "")
    matches = []
    for m in AVAILABLE_MODELS:
        mid = m["id"].lower().replace("-", "").replace("_", "")
        mname = m["name"].lower().replace("-", "").replace("_", "")
        if query_lower in mid or query_lower in mname:
            matches.append(m)
    return matches


def get_hardware_compatible_models(available_vram_gb: float) -> list[dict[str, Any]]:
    compatible = []
    for m in AVAILABLE_MODELS:
        if m["min_vram_gb"] <= available_vram_gb * 0.85:
            compatible.append(m)
    return compatible
