import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from bookextractor.exceptions import ModelNotAvailableError, ParsingError


def test_model_not_available_error_default_message():
    err = ModelNotAvailableError()
    assert "load_vlm" in err.message


def test_model_not_available_error_custom_message():
    err = ModelNotAvailableError("custom message")
    assert err.message == "custom message"


def test_parsing_error_default_message():
    err = ParsingError()
    assert err.message == "Failed to parse LLM output."


def test_parsing_error_custom_message():
    err = ParsingError("custom parse error")
    assert err.message == "custom parse error"


def test_parsing_error_raw_output():
    err = ParsingError("bad output", raw_output='{"invalid": ')
    assert err.raw_output == '{"invalid": '


def test_parsing_error_no_raw_output():
    err = ParsingError("bad output")
    assert err.raw_output is None


def test_exceptions_inherit_from_base():
    assert issubclass(ModelNotAvailableError, Exception)
    assert issubclass(ParsingError, Exception)


def test_model_manager_singleton():
    from bookextractor.model_manager import ModelManager

    ModelManager.reset()
    manager1 = ModelManager.get_instance()
    manager2 = ModelManager.get_instance()
    assert manager1 is manager2
    ModelManager.reset()


def test_model_manager_not_loaded_initially():
    from bookextractor.model_manager import ModelManager

    ModelManager.reset()
    manager = ModelManager()
    assert manager.is_loaded() is False
    assert manager.get_model() is None


def test_model_manager_is_loaded_with_model():
    from bookextractor.model_manager import ModelManager

    manager = ModelManager()
    manager._model = MagicMock()
    assert manager.is_loaded() is True
    assert manager.get_model() is not None


def test_model_manager_model_id():
    from bookextractor.model_manager import ModelManager

    manager = ModelManager()
    assert manager.model_id is None
    manager._model_id = "test-model"
    assert manager.model_id == "test-model"


def test_model_manager_thread_safety():
    from bookextractor.model_manager import ModelManager

    ModelManager.reset()
    results = []
    errors = []

    def get_manager():
        try:
            manager = ModelManager.get_instance()
            results.append(manager)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=get_manager) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 10
    # All instances should be the same
    assert all(r is results[0] for r in results)
    ModelManager.reset()


def test_vllm_target_device_env_guard(monkeypatch):
    from unittest.mock import patch

    from bookextractor.model_manager import ModelManager

    ModelManager.reset()

    monkeypatch.setenv("VLLM_TARGET_DEVICE", "custom_tpu_device")

    config = {"device": "cuda", "dtype": "float16", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}

    class FakeLLM:
        __name__ = "LLM"

        def __init__(self, **_kwargs):
            pass

    with (
        patch("bookextractor.hardware.get_vllm_config", return_value=config),
        patch("bookextractor.model_manager.LLM", FakeLLM),
    ):
        manager = ModelManager()
        manager.get_or_create(model_id="test-model", max_model_len=4096)

        assert os.environ["VLLM_TARGET_DEVICE"] == "custom_tpu_device"

    ModelManager.reset()