# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
Model detection and configuration for MinerU backend.
"""

import os
from pathlib import Path
from typing import Optional


MINERU_MODELS_DIR_ENV = "LRMCP_MINERU_MODELS_DIR"
DEFAULT_MINERU_MODELS_DIR = "~/.cache/local_read_mcp/models"


class ModelDetector:
    """Detect available models and backends."""

    def __init__(self):
        self.mineru_models_dir = self._get_mineru_models_dir()

    def _get_mineru_models_dir(self) -> Path:
        """Get MinerU models directory from environment or default."""
        dir_path = os.environ.get(MINERU_MODELS_DIR_ENV, DEFAULT_MINERU_MODELS_DIR)
        return Path(dir_path).expanduser()

    @property
    def mineru_available(self) -> bool:
        """Check if MinerU models are available."""
        if not self.mineru_models_dir.exists():
            return False
        # Check for at least some model files
        # This is a simplified check - real implementation would check for specific models
        return any(self.mineru_models_dir.iterdir())

    @property
    def mineru_warning(self) -> Optional[str]:
        """Get warning message for MinerU backend."""
        if self.mineru_available:
            return None
        return (
            "MinerU models not found. To use MinerU backend:\n"
            f"  1. Download models from: https://github.com/opendatalab/MinerU#model-download\n"
            f"  2. Extract to: {self.mineru_models_dir}\n"
            f"  Or set {MINERU_MODELS_DIR_ENV} environment variable\n"
            f"Falling back to Simple backend."
        )

    @property
    def vlm_available(self) -> bool:
        """Check if VLM API is configured."""
        from ..config import get_config
        config = get_config()
        return bool(config.api_key)

    @property
    def vlm_warning(self) -> Optional[str]:
        """Get warning message for VLM backends."""
        if self.vlm_available:
            return None
        return (
            "VLM API not configured. To use VLM backends:\n"
            "  1. Set VISION_API_KEY (or OPENAI_API_KEY) environment variable\n"
            "  2. Optionally set VISION_BASE_URL and VISION_MODEL\n"
            "Falling back to Simple backend."
        )


# Global detector instance
_detector: Optional[ModelDetector] = None


def get_model_detector() -> ModelDetector:
    """Get the global model detector."""
    global _detector
    if _detector is None:
        _detector = ModelDetector()
    return _detector
