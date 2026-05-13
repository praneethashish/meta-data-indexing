import os
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from bookextractor.hardware import _check_mps, _check_nvidia_nvml, _check_nvidia_torch, get_vllm_config


@pytest.fixture
def mock_pynvml():
    with patch("bookextractor.hardware.pynvml") as mock:
        yield mock


def test_check_tpu_mock():
    # Cover the TPU branch in get_vllm_config
    mock_res = {"device": "tpu", "count": 1, "memory_gb": 16.0, "name": "TPU", "precision_supported": ["bfloat16"]}
    with patch("bookextractor.hardware._check_tpu", return_value=mock_res):
        res = get_vllm_config()
        assert res["device"] == "tpu"


def test_check_nvidia_nvml_error(mock_pynvml):
    mock_pynvml.nvmlInit.side_effect = Exception("error")
    assert _check_nvidia_nvml() is None


def test_check_nvidia_nvml_bytes_name(mock_pynvml):
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle"
    mock_memory = MagicMock()
    mock_memory.total = 16 * 1024**3
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory
    mock_pynvml.nvmlDeviceGetName.return_value = b"Tesla T4"
    res = _check_nvidia_nvml()
    assert res is not None
    assert res["name"] == "Tesla T4"


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_check_nvidia_torch_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        assert _check_nvidia_torch() is None


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_check_mps_no_mps():
    with patch("torch.backends.mps.is_available", return_value=False):
        assert _check_mps() is None


def test_get_vllm_config_cuda_branches():
    # Test V100 branch
    hw = {
        "device": "cuda",
        "name": "Tesla V100",
        "memory_gb": 32.0,
        "count": 1,
        "precision_supported": ["float16", "bfloat16"],
    }
    with patch("bookextractor.hardware.detect_hardware", return_value=hw):
        config = get_vllm_config()
        assert config["dtype"] == "bfloat16"

    # Test small memory branch
    hw["memory_gb"] = 4.0
    with patch("bookextractor.hardware.detect_hardware", return_value=hw):
        config = get_vllm_config()
        assert config["dtype"] == "float16"


def test_get_vllm_config_overrides():
    with patch.dict(
        os.environ,
        {
            "VLLM_TENSOR_PARALLEL_SIZE": "auto",
            "VLLM_DEVICE": "cuda",
            "VLLM_DTYPE": "bfloat16",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.5",
        },
    ):
        hw = {"device": "cuda", "count": 4, "name": "GPU", "memory_gb": 16.0, "precision_supported": ["float16"]}
        with patch("bookextractor.hardware.detect_hardware", return_value=hw):
            config = get_vllm_config()
            assert config["tensor_parallel_size"] == 4
            assert config["dtype"] == "bfloat16"
            assert config["gpu_memory_utilization"] == 0.5
