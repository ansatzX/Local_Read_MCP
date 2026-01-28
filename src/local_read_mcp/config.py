# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
Configuration management for Local Read MCP server.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global config holder
_config = None


class Config:
    """Configuration management for Local Read MCP server."""

    def __init__(self, dotenv_path: Optional[Path] = None):
        """
        Initialize configuration from .env file.

        Args:
            dotenv_path: Path to .env file (only used on first call or if reload=True)
        """
        self.dotenv_path = dotenv_path or Path.cwd()

        # Initialize settings
        self._init_settings()

    def _init_settings(self):
        """Initialize settings from environment variables."""
        # Vision API settings (for OpenAI-compatible APIs like Doubao)
        # Uses simple naming as requested: api_key, base_url, model
        self.api_key = os.environ.get("VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("VISION_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.model = os.environ.get("VISION_MODEL") or os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")

        # Vision settings
        self.vision_max_image_size_mb = int(os.environ.get("VISION_MAX_IMAGE_SIZE_MB", "20"))

        # PDF processing settings
        self.pdf_extract_images_default = os.environ.get("PDF_EXTRACT_IMAGES_DEFAULT", "false").lower() == "true"
        self.pdf_images_output_dir = os.environ.get("PDF_IMAGES_OUTPUT_DIR", "/tmp/local_read_mcp_images")

        # Logging
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Determine if vision features are enabled
        self._vision_enabled = self._check_vision_enabled()

    def _check_vision_enabled(self) -> bool:
        """
        Check if vision features are enabled based on configuration.

        Vision is enabled if API_KEY is set.
        """
        return bool(self.api_key)

    @property
    def vision_enabled(self) -> bool:
        """Get whether vision features are enabled."""
        return self._vision_enabled

    def __repr__(self):
        """String representation of config (hide sensitive data)."""
        return (
            f"Config(\n"
            f"  dotenv_path={self.dotenv_path},\n"
            f"  api_key={'***' if self.api_key else 'Not Set'},\n"
            f"  base_url={self.base_url},\n"
            f"  model={self.model},\n"
            f"  vision_enabled={self.vision_enabled},\n"
            f"  pdf_extract_images_default={self.pdf_extract_images_default}\n"
            f")"
        )


def get_config(dotenv_path: Optional[Path] = None, reload: bool = False) -> Config:
    """
    Get global configuration instance.

    Args:
        dotenv_path: Path to .env file (only used on first call or if reload=True)
        reload: Force reload configuration from .env file

    Returns:
        Config instance
    """
    global _config
    if _config is None or reload:
        _config = Config(dotenv_path=dotenv_path)
    return _config
