"""
Configuration management for Local Read MCP.

Handles loading environment variables from .env file with support for:
- Custom .env file path via parameter
- Default .env location at repository root
- Fallback to system environment variables
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Global config holder
_config = None


class Config:
    """Configuration container for Local Read MCP."""

    def __init__(self, dotenv_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            dotenv_path: Path to .env file. If None, uses default location (repo root/.env)
        """
        self.dotenv_path = self._resolve_dotenv_path(dotenv_path)
        self._load_env()
        self._init_settings()

    def _resolve_dotenv_path(self, dotenv_path: Optional[str]) -> Path:
        """
        Resolve .env file path.

        Args:
            dotenv_path: Custom path or None for default

        Returns:
            Path to .env file
        """
        if dotenv_path:
            return Path(dotenv_path).resolve()

        # Default: repository root/.env
        # This file is in src/local_read_mcp/config.py
        # Repository root is 2 levels up
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / ".env"

    def _load_env(self):
        """Load environment variables from .env file if it exists."""
        if self.dotenv_path.exists():
            logger.info(f"Loading environment variables from: {self.dotenv_path}")
            # Manual .env parsing (avoid external dependency on python-dotenv)
            with open(self.dotenv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        # Set environment variable (only if not already set)
                        if key not in os.environ:
                            os.environ[key] = value
        else:
            logger.warning(f".env file not found at: {self.dotenv_path}")
            logger.info("Using system environment variables only")

    def _init_settings(self):
        """Initialize settings from environment variables."""
        # Vision API settings (for vision_server.py)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")

        # Ollama settings (local vision model)
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.environ.get("OLLAMA_VISION_MODEL", "llava:13b")

        # Vision settings
        self.vision_default_provider = os.environ.get("VISION_DEFAULT_PROVIDER", "none")  # "none", "ollama", "openai"
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

    def __repr__(self):
        """String representation of config (hide sensitive data)."""
        return (
            f"Config(\n"
            f"  dotenv_path={self.dotenv_path},\n"
            f"  openai_api_key={'***' if self.openai_api_key else 'Not Set'},\n"
            f"  openai_base_url={self.openai_base_url},\n"
            f"  ollama_base_url={self.ollama_base_url},\n"
            f"  vision_default_provider={self.vision_default_provider},\n"
            f"  pdf_extract_images_default={self.pdf_extract_images_default}\n"
            f")"
        )


def get_config(dotenv_path: Optional[str] = None, reload: bool = False) -> Config:
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
