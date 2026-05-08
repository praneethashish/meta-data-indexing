import logging
import os
from typing import Any

from vllm import LLM, SamplingParams

from .hardware import get_vllm_config

logger = logging.getLogger(__name__)


class VLMClient:
    """
    Unified wrapper for Gemma/Qwen-VL vLLM inference.
    Handles hardware-optimized initialization and provides methods for text and vision tasks.
    """

    _instance: "VLMClient | None" = None
    _llm: LLM | None = None

    def __init__(self, model_id: str | None = None):
        """
        Initialize the VLMClient. Note: Use get_instance() for shared LLM resource.
        """
        if VLMClient._llm is None:
            self.model_id = model_id or os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
            self._initialize_llm()

    @classmethod
    def get_instance(cls, model_id: str | None = None) -> "VLMClient":
        """Get or create a singleton instance of VLMClient."""
        if cls._instance is None:
            cls._instance = cls(model_id=model_id)
        return cls._instance

    def _initialize_llm(self) -> None:
        """Initialize vLLM engine with hardware-optimized configuration."""
        config = get_vllm_config()

        logger.info(f"Initializing VLMClient with hardware: {config['detected_hardware']['name']}")
        logger.info(
            f"Applying vLLM config: dtype={config['dtype']}, "
            f"tp_size={config['tensor_parallel_size']}, "
            f"memory_util={config['gpu_memory_utilization']}"
        )

        # Set environment variable to force device type (fixes vLLM auto-detection issues)
        os.environ["VLLM_TARGET_DEVICE"] = config["device"]

        VLMClient._llm = LLM(
            model=self.model_id,
            tensor_parallel_size=config["tensor_parallel_size"],
            dtype=config["dtype"],
            max_model_len=8192,
            gpu_memory_utilization=config["gpu_memory_utilization"],
            trust_remote_code=True,
        )

    def generate(self, prompts: list[str], sampling_params: SamplingParams | None = None) -> list[Any]:
        """Generate text from one or more prompts."""
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

        if VLMClient._llm is None:
            raise RuntimeError("vLLM engine not initialized")

        return VLMClient._llm.generate(prompts, sampling_params)

    async def describe_image(self, image_path: str) -> dict[str, Any]:
        """
        Generate description for an image.
        Note: Currently a placeholder until full Phase 2 vision processing is implemented.
        """
        # In a future update, this will handle image encoding and multi-modal prompt generation
        # for models like Qwen2-VL or Gemma-VL.
        _ = image_path  # Silence unused argument warning
        return {
            "description": "Image description not yet implemented in VLMClient wrapper.",
            "scene_classification": "other",
            "text_content": None,
        }
