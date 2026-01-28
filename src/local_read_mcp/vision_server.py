"""
Vision MCP Server for Local Read MCP.

Provides visual question answering capabilities using OpenAI-compatible APIs
(e.g., OpenAI GPT-4o, Doubao, etc.).
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from fastmcp import FastMCP

# Import config management
from .config import get_config
from .vision.client import call_openai_vision, check_vision_available

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("local_read_vision_server")

# Global config will be initialized when server starts
config = None


def guess_mime_type_from_extension(file_path: str) -> str:
    """
    Guess the MIME type based on the file extension.

    Args:
        file_path: Path to the image file

    Returns:
        MIME type string (e.g., "image/jpeg")
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }

    return mime_types.get(ext, "image/jpeg")  # Default to JPEG


@mcp.tool()
async def analyze_image(
    image_path: str,
    question: str = "Describe this image in detail. What type of content is it (chart, diagram, photo, etc.)?",
    api_key: Optional[str] = None
) -> str:
    """Analyze an image and answer questions about it.

    This tool uses OpenAI-compatible APIs (e.g., GPT-4o, Doubao).

    Args:
        image_path: Path to the image file to analyze
        question: Question to ask about the image (default: general description)
        api_key: API key (only needed if not set in .env file)

    Returns:
        str: Answer to the question about the image

    Examples:
        # Use API key from .env file
        analyze_image("/path/to/chart.png", "What data does this chart show?")

        # Override API key
        analyze_image(
            "/path/to/diagram.png",
            "Explain this system architecture diagram",
            api_key="sk-..."
        )

    Environment Variables (.env):
        OPENAI_API_KEY: Your API key (required)
        OPENAI_BASE_URL: API base URL (default: https://api.openai.com/v1)
        OPENAI_VISION_MODEL: Model name (default: gpt-4o)
        VISION_MAX_IMAGE_SIZE_MB: Maximum image size (default: 20)
    """
    global config
    if config is None:
        config = get_config()

    # Validate image path
    if not os.path.exists(image_path):
        return f"[ERROR]: Image file not found: {image_path}"

    # Check file size
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > config.vision_max_image_size_mb:
        return (
            f"[ERROR]: Image file too large ({file_size_mb:.2f}MB). "
            f"Maximum allowed: {config.vision_max_image_size_mb}MB"
        )

    # Use provided API key or fallback to config
    effective_api_key = api_key or config.openai_api_key

    if not effective_api_key:
        return (
            "[ERROR]: API key not provided. "
            "Please provide api_key parameter or set OPENAI_API_KEY in .env file."
        )

    logger.info(f"Using vision API at {config.openai_base_url} with model {config.openai_model}")
    return await call_openai_vision(
        image_path=image_path,
        question=question,
        api_key=effective_api_key,
        base_url=config.openai_base_url,
        model=config.openai_model
    )


@mcp.tool()
async def get_vision_status() -> Dict[str, Any]:
    """Get vision server status and configuration.

    Returns information about available vision API and current configuration.

    Returns:
        dict: Status information including:
            - available: Whether vision API is configured
            - openai_configured: Whether API key is set
            - message: Human-readable status message
            - suggestion: Configuration suggestion if not available
    """
    global config
    if config is None:
        config = get_config()

    return check_vision_available(config)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Vision MCP Server for Local Read MCP")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport method: 'stdio' or 'http' (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port to use when running with HTTP transport (default: 8081)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/vision",
        help="URL path to use when running with HTTP transport (default: /vision)",
    )
    parser.add_argument(
        "--dotenv",
        type=str,
        default=None,
        help="Path to .env file (default: repository root/.env)",
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Initialize config with custom .env path if provided
    config = get_config(dotenv_path=args.dotenv)
    logger.info(f"Vision Server Configuration:\n{config}")

    # Run the server with the specified transport method
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # For HTTP transport, include port and path options
        mcp.run(transport="streamable-http", port=args.port, path=args.path)
