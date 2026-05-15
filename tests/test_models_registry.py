import tempfile
from pathlib import Path

import pytest

from bookextractor.models_registry import (
    AVAILABLE_MODELS,
    cache_dir_for_model,
    find_model_by_query,
    get_cached_models,
    get_hardware_compatible_models,
    is_model_cached,
)


class TestModelsRegistry:
    def test_available_models_defined(self):
        assert len(AVAILABLE_MODELS) == 6
        ids = [m["id"] for m in AVAILABLE_MODELS]
        assert "Qwen/Qwen2.5-VL-7B-Instruct" in ids
        assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in ids
        assert "Qwen/Qwen3.5-9B" in ids
        assert "google/gemma-3-27b-it" in ids
        assert "google/gemma-4-31b-it" in ids

    def test_cache_dir_for_model(self):
        path = cache_dir_for_model("Qwen/Qwen2.5-VL-7B-Instruct")
        assert "models--Qwen--Qwen2__dot__5-VL-7B-Instruct" in str(path)

    def test_cache_dir_with_dots(self):
        path = cache_dir_for_model("google/gemma-3-27b-it")
        assert "models--google--gemma-3-27b-it" in str(path)

    def test_is_model_cached_nonexistent(self):
        assert is_model_cached("nonexistent/model") is False

    def test_get_cached_models_empty(self):
        cached = get_cached_models()
        assert isinstance(cached, list)

    @pytest.fixture
    def fake_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import bookextractor.models_registry as mr

            original = mr.HF_CACHE_DIR
            mr.HF_CACHE_DIR = Path(tmpdir)

            model_id = "testorg/testmodel"
            cache_path = mr.cache_dir_for_model(model_id)
            cache_path.mkdir(parents=True)
            snapshots = cache_path / "snapshots" / "abc123"
            snapshots.mkdir(parents=True)
            f = snapshots / "model.safetensors"
            f.write_text("fake")
            yield model_id, tmpdir

            mr.HF_CACHE_DIR = original

    def test_is_model_cached_with_cache(self, fake_cache):
        model_id, _ = fake_cache
        assert is_model_cached(model_id) is True

    def test_get_cached_models_with_cache(self, fake_cache):
        from bookextractor.models_registry import get_cached_models as gcm

        model_id, _ = fake_cache
        cached = gcm()
        assert model_id in cached

    def test_get_cached_model_size(self, fake_cache):
        from bookextractor.models_registry import get_cached_model_size as gcms

        model_id, _ = fake_cache
        size = gcms(model_id)
        assert size > 0

    def test_remove_model_from_cache(self, fake_cache):
        from bookextractor.models_registry import remove_model_from_cache as rmfc

        model_id, _ = fake_cache
        assert rmfc(model_id) is True
        assert rmfc(model_id) is False

    def test_find_model_by_query_full_id(self):
        matches = find_model_by_query("Qwen/Qwen3-VL-30B-A3B-Instruct")
        assert len(matches) == 1
        assert matches[0]["id"] == "Qwen/Qwen3-VL-30B-A3B-Instruct"

    def test_find_model_by_query_fuzzy(self):
        matches = find_model_by_query("gemma-3")
        assert len(matches) >= 1
        ids = [m["id"] for m in matches]
        assert "google/gemma-3-27b-it" in ids

    def test_find_model_by_query_partial(self):
        matches = find_model_by_query("gemma")
        assert len(matches) == 3

    def test_find_model_by_query_no_match(self):
        matches = find_model_by_query("nonexistent")
        assert matches == []

    def test_get_hardware_compatible_models(self):
        compatible = get_hardware_compatible_models(32)
        ids = [m["id"] for m in compatible]
        assert "Qwen/Qwen2.5-VL-7B-Instruct" in ids
        assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in ids
        assert "Qwen/Qwen3.5-9B" in ids

        compatible = get_hardware_compatible_models(8)
        ids = [m["id"] for m in compatible]
        assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in ids

    def test_has_desc_for_all_models(self):
        for m in AVAILABLE_MODELS:
            assert "desc" in m, f"Missing desc for {m['id']}"
            assert len(m["desc"]) > 0

    def test_has_min_vram_for_all_models(self):
        for m in AVAILABLE_MODELS:
            assert m["min_vram_gb"] > 0, f"Invalid min_vram for {m['id']}"


class TestModelManagerAutoDetect:
    def test_get_or_create_auto_detects_when_no_model_id(self):
        from unittest.mock import patch

        from bookextractor.model_manager import ModelManager

        ModelManager.reset()

        class FakeLLM:
            __name__ = "LLM"

            def __init__(self, **_kwargs):
                pass

        mock_config = {
            "device": "cuda",
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
        }

        with (
            patch("bookextractor.hardware.get_vllm_config", return_value=mock_config),
            patch("bookextractor.model_manager.LLM", FakeLLM),
            patch("bookextractor.models_registry.get_cached_models", return_value=[]),
        ):
            manager = ModelManager.get_instance()
            manager.get_or_create(model_id=None, max_model_len=4096)
            assert manager.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
            assert manager.is_loaded()

        ModelManager.reset()
