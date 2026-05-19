import logging
import os
from typing import Any

import psutil

from .config import settings

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _check_tpu() -> dict[str, Any] | None:
    """Helper to detect TPU hardware."""
    try:
        import torch_xla.core.xla_model as xm

        devices = xm.get_xla_supported_devices()
        if devices:
            return {
                "device": "tpu",
                "count": len(devices),
                "memory_gb": 16.0,  # Common for v3
                "name": "Google TPU",
                "precision_supported": ["bfloat16", "float32"],
            }
    except ImportError:
        pass
    return None


def _check_nvidia_nvml() -> dict[str, Any] | None:
    """Helper to detect NVIDIA GPU via NVML."""
    if not pynvml:
        return None

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            hw = {
                "device": "cuda",
                "count": device_count,
                "memory_gb": info.total / (1024**3),
                "name": name,
                "precision_supported": ["float32"],
            }

            if torch and torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                if major >= 8:
                    hw["precision_supported"].append("bfloat16")
                hw["precision_supported"].append("float16")

            return hw
    except Exception as e:
        logger.warning(f"pynvml initialization failed: {e}. Falling back to torch.cuda.")
    finally:
        try:
            if pynvml:
                pynvml.nvmlShutdown()
        except Exception:  # nosec B110
            pass
    return None


def _check_nvidia_torch() -> dict[str, Any] | None:
    """Helper to detect NVIDIA GPU via Torch."""
    if torch and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        precision = ["float32", "float16"]
        if major >= 8:
            precision.append("bfloat16")

        return {
            "device": "cuda",
            "count": torch.cuda.device_count(),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "name": torch.cuda.get_device_name(0),
            "precision_supported": precision,
        }
    return None


def _check_mps() -> dict[str, Any] | None:
    """Helper to detect Apple Silicon MPS."""
    if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {
            "device": "mps",
            "count": 1,
            "memory_gb": psutil.virtual_memory().total / (1024**3),  # Unified memory
            "name": "Apple Silicon MPS",
            "precision_supported": ["float16", "float32"],
        }
    return None


def detect_hardware() -> dict[str, Any]:
    """
    Detects available hardware accelerators and system memory.
    Returns a dictionary with hardware details.
    """
    # 1. Check for TPU (Priority in environments like Colab)
    tpu = _check_tpu()
    if tpu:
        return tpu

    # 2. Check for NVIDIA GPU
    # In Colab, torch.cuda.is_available() is often more reliable than NVML library paths
    gpu = _check_nvidia_torch() or _check_nvidia_nvml()
    if gpu:
        return gpu

    # 3. Check for Apple Silicon MPS
    mps = _check_mps()
    if mps:
        return mps

    # 4. CPU Fallback
    return {
        "device": "cpu",
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "count": 0,
        "name": "CPU",
        "precision_supported": ["float32"],
    }


def _get_cuda_config(hw: dict[str, Any]) -> tuple[str, float, int]:
    """Helper to map CUDA hardware to vLLM config."""
    name = hw["name"].upper()
    memory = hw["memory_gb"]
    count = hw["count"]
    precision = hw["precision_supported"]

    dtype = "float32"
    util = 0.85
    tp = 1

    if memory < 8:
        dtype = "float16"
        util = 0.60
    elif "T4" in name:
        dtype = "float16"
        util = 0.70
    elif "V100" in name:
        dtype = "bfloat16" if "bfloat16" in precision else "float16"
        util = 0.85
    elif "A100" in name:
        dtype = "bfloat16"
        util = 0.90
        tp = count
    elif "RTX 3090" in name or "RTX 4090" in name:
        dtype = "bfloat16"
        util = 0.85
    else:
        dtype = "bfloat16" if "bfloat16" in precision else "float16"
        util = 0.85

    return dtype, util, tp


def get_vllm_config() -> dict[str, Any]:
    """
    Generates optimized vLLM configuration based on detected hardware.
    Supports overrides via environment variables.
    """
    hw = detect_hardware()
    device = hw["device"]
    memory = hw["memory_gb"]

    # Defaults
    dtype = "float32"
    util = 0.85
    tp = 1

    if device == "tpu":
        dtype = "bfloat16"
        util = 0.80 if memory < 16 else 0.85
        tp = 1 if memory < 16 else hw["count"]
    elif device == "cuda":
        dtype, util, tp = _get_cuda_config(hw)
    elif device == "mps":
        dtype = "float16"
        util = 0.75

    # Overrides
    device = settings.VLLM_DEVICE or os.getenv("VLLM_DEVICE", device)
    dtype = settings.VLLM_DTYPE or os.getenv("VLLM_DTYPE", dtype)
    util = settings.VLLM_GPU_MEMORY_UTILIZATION or float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", util))

    env_tp = os.getenv("VLLM_TENSOR_PARALLEL_SIZE")
    if settings.VLLM_TENSOR_PARALLEL_SIZE:
        tp = settings.VLLM_TENSOR_PARALLEL_SIZE
    elif env_tp:
        tp = hw["count"] if env_tp.lower() == "auto" and device != "cpu" else int(env_tp)

    return {
        "device": device,
        "dtype": dtype,
        "gpu_memory_utilization": util,
        "tensor_parallel_size": tp,
        "detected_hardware": hw,
    }
