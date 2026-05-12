import os
from unittest.mock import MagicMock, patch

import pytest

from bookextractor.hardware import detect_hardware, get_vllm_config


@pytest.fixture
def mock_pynvml():
    with patch("bookextractor.hardware.pynvml") as mock:
        yield mock


@pytest.fixture
def mock_torch():
    with patch("bookextractor.hardware.torch") as mock:
        yield mock


@pytest.fixture
def mock_psutil():
    with patch("bookextractor.hardware.psutil") as mock:
        mock.virtual_memory.return_value.total = 32 * (1024**3)
        yield mock


def test_cpu_fallback_detection(mock_pynvml, mock_torch):
    """Test fallback to CPU when no accelerators are found."""
    mock_pynvml.nvmlInit.side_effect = Exception("No GPU")
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False

    with patch("builtins.__import__", side_effect=ImportError):
        hw = detect_hardware()

    assert hw["device"] == "cpu"
    assert hw["name"] == "CPU"


def test_nvidia_gpu_detection(mock_pynvml, mock_torch):
    """Test detection of NVIDIA GPU."""
    _ = mock_pynvml  # Silence unused argument warning
    # Since torch.cuda.is_available() is checked first now
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_props = MagicMock()
    mock_props.total_memory = 16 * (1024**3)
    mock_torch.cuda.get_device_properties.return_value = mock_props
    mock_torch.cuda.get_device_name.return_value = "Tesla T4"
    mock_torch.cuda.get_device_capability.return_value = (7, 5)

    hw = detect_hardware()

    assert hw["device"] == "cuda"
    assert hw["memory_gb"] == 16.0
    assert "Tesla T4" in hw["name"]
    assert "float16" in hw["precision_supported"]
    assert "bfloat16" not in hw["precision_supported"]


def test_high_memory_gpu_config(mock_pynvml, mock_torch):
    """Test config generation for high-memory GPU (A100 style)."""
    _ = mock_pynvml  # Silence unused argument warning
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_props = MagicMock()
    mock_props.total_memory = 80 * (1024**3)
    mock_torch.cuda.get_device_properties.return_value = mock_props
    mock_torch.cuda.get_device_name.return_value = "NVIDIA A100-SXM4-80GB"
    mock_torch.cuda.get_device_capability.return_value = (8, 0)

    config = get_vllm_config()

    assert config["dtype"] == "bfloat16"
    assert config["gpu_memory_utilization"] == 0.90
    assert config["tensor_parallel_size"] == 1  # 1 GPU


def test_mps_detection(mock_pynvml, mock_torch):
    """Test Apple Silicon MPS detection."""
    mock_pynvml.nvmlInit.side_effect = Exception()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True

    hw = detect_hardware()

    assert hw["device"] == "mps"
    assert hw["name"] == "Apple Silicon MPS"

    config = get_vllm_config()
    assert config["dtype"] == "float16"
    assert config["gpu_memory_utilization"] == 0.75


def test_env_var_overrides():
    """Test that environment variables override auto-detection."""
    with patch.dict(
        os.environ,
        {
            "VLLM_DEVICE": "tpu",
            "VLLM_DTYPE": "float32",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.5",
            "VLLM_TENSOR_PARALLEL_SIZE": "4",
        },
    ):
        config = get_vllm_config()

    assert config["device"] == "tpu"
    assert config["dtype"] == "float32"
    assert config["gpu_memory_utilization"] == 0.5
    assert config["tensor_parallel_size"] == 4
