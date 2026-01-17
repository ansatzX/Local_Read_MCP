"""
Vision MCP Server for Local Read MCP.

Provides visual question answering capabilities using:
- Local models via Ollama (default, no API key needed)
- OpenAI GPT-4o Vision API (requires API key)

This is a separate MCP server that can be configured independently.
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


def check_vision_availability() -> Dict[str, Any]:
    """
    Check if vision analysis is available and which providers are configured.

    Returns:
        Dict with:
        - available: bool - Whether any vision provider is available
        - openai_configured: bool - Whether OpenAI/Doubao API is configured
        - ollama_configured: bool - Whether Ollama is configured (assumed available if URL is set)
        - message: str - Human-readable status message
        - suggestion: str - Suggestion for configuration if not available
    """
    from .config import get_config
    config = get_config()

    openai_configured = bool(config.openai_api_key)
    ollama_configured = True  # Ollama assumed available if user wants to use it

    result = {
        "available": openai_configured or ollama_configured,
        "openai_configured": openai_configured,
        "ollama_configured": ollama_configured,
        "providers": []
    }

    if openai_configured:
        result["providers"].append("openai")
    if ollama_configured:
        result["providers"].append("ollama")

    if not openai_configured and not ollama_configured:
        result["message"] = "视觉分析功能未配置"
        result["suggestion"] = (
            "要启用视觉分析功能，请在.env文件中配置以下选项之一：\n"
            "1. 云端API (豆包/OpenAI): 设置 OPENAI_API_KEY 和 OPENAI_BASE_URL\n"
            "2. 本地模型 (Ollama): 安装并运行 Ollama，然后设置 VISION_DEFAULT_PROVIDER=ollama"
        )
    elif openai_configured:
        result["message"] = f"OpenAI/豆包 API 已配置 (模型: {config.openai_model})"
        result["suggestion"] = "视觉分析功能可用"
    else:
        result["message"] = f"Ollama 本地模型可用 (模型: {config.ollama_model})"
        result["suggestion"] = "视觉分析功能可用（本地模式）"

    return result


async def call_openai_vision(
    image_path: str,
    question: str,
    api_key: str,
    base_url: str,
    model: str
) -> str:
    """
    Call OpenAI GPT-4o Vision API.

    Args:
        image_path: Path to image file
        question: Question about the image
        api_key: OpenAI API key
        base_url: OpenAI API base URL
        model: Model name (e.g., "gpt-4o")

    Returns:
        Answer from the model
    """
    # Early check: API key must be provided
    if not api_key:
        return (
            "[VISION_NOT_CONFIGURED]: 视觉分析功能未配置。\n"
            "原因: 缺少 OPENAI_API_KEY 配置。\n"
            "解决方案: 在 .env 文件中设置 OPENAI_API_KEY=your-api-key\n"
            "提示: 支持豆包API，将 OPENAI_BASE_URL 设置为豆包的API地址即可。"
        )

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return "[ERROR]: openai package not installed. Install with: pip install openai"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    mime_type = guess_mime_type_from_extension(image_path)

    # Build message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                }
            ]
        }
    ]

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"[ERROR]: OpenAI API call failed: {str(e)}"


async def call_ollama_vision(
    image_path: str,
    question: str,
    base_url: str,
    model: str
) -> str:
    """
    Call Ollama vision model (e.g., LLaVA).

    Args:
        image_path: Path to image file
        question: Question about the image
        base_url: Ollama API base URL
        model: Model name (e.g., "llava:13b")

    Returns:
        Answer from the model
    """
    try:
        import aiohttp
    except ImportError:
        return "[ERROR]: aiohttp package not installed. Install with: pip install aiohttp"

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Ollama API format
    payload = {
        "model": model,
        "prompt": question,
        "images": [image_data],
        "stream": False
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "[ERROR]: No response from Ollama")
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {error_text}")
                    return f"[ERROR]: Ollama API call failed (status {response.status}): {error_text}"

    except Exception as e:
        logger.error(f"Ollama connection error: {e}")
        return f"[ERROR]: Failed to connect to Ollama at {base_url}: {str(e)}"


@mcp.tool()
async def analyze_image(
    image_path: str,
    question: str = "Describe this image in detail. What type of content is it (chart, diagram, photo, etc.)?",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """Analyze an image and answer questions about it.

    This tool supports both local vision models (via Ollama) and cloud APIs (OpenAI GPT-4o).

    Args:
        image_path: Path to the image file to analyze
        question: Question to ask about the image (default: general description)
        model: Vision model to use. Options:
            - "ollama" or "local": Use local Ollama (default, no API key needed)
            - "openai" or "gpt-4o": Use OpenAI GPT-4o (requires API key)
            - If not specified, uses VISION_DEFAULT_PROVIDER from .env
        api_key: OpenAI API key (only needed if model="openai" and not set in .env)

    Returns:
        str: Answer to the question about the image

    Examples:
        # Use local Ollama (default)
        analyze_image("/path/to/chart.png", "What data does this chart show?")

        # Use OpenAI GPT-4o
        analyze_image(
            "/path/to/diagram.png",
            "Explain this system architecture diagram",
            model="openai",
            api_key="sk-..."
        )

    Environment Variables (.env):
        VISION_DEFAULT_PROVIDER: "none", "ollama", or "openai" (default: "none")
        OLLAMA_BASE_URL: Ollama API URL (default: http://localhost:11434)
        OLLAMA_VISION_MODEL: Ollama model name (default: llava:13b)
        OPENAI_API_KEY: OpenAI API key
        OPENAI_BASE_URL: OpenAI API URL (default: https://api.openai.com/v1)
        OPENAI_VISION_MODEL: OpenAI model name (default: gpt-4o)
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

    # Determine which model to use
    if model is None:
        model = config.vision_default_provider

    model_lower = model.lower()

    # Route to appropriate backend
    if model_lower in ["none", "disabled"]:
        return (
            "[INFO]: Vision analysis is disabled. "
            "Set model='ollama' or model='openai' to enable, "
            "or configure VISION_DEFAULT_PROVIDER in .env file."
        )

    elif model_lower in ["ollama", "local"]:
        logger.info(f"Using Ollama vision model: {config.ollama_model}")
        return await call_ollama_vision(
            image_path=image_path,
            question=question,
            base_url=config.ollama_base_url,
            model=config.ollama_model
        )

    elif model_lower in ["openai", "gpt-4o", "gpt4o"]:
        # Use provided API key or fallback to config
        effective_api_key = api_key or config.openai_api_key

        if not effective_api_key:
            return (
                "[ERROR]: OpenAI API key not provided. "
                "Please provide api_key parameter or set OPENAI_API_KEY in .env file."
            )

        logger.info(f"Using OpenAI vision model: {config.openai_model}")
        return await call_openai_vision(
            image_path=image_path,
            question=question,
            api_key=effective_api_key,
            base_url=config.openai_base_url,
            model=config.openai_model
        )

    else:
        return (
            f"[ERROR]: Unknown model '{model}'. "
            f"Valid options: 'ollama' (local), 'openai' (cloud), or 'none' (disabled)"
        )


@mcp.tool()
async def get_vision_status() -> Dict[str, Any]:
    """Get vision server status and configuration.

    Returns information about available vision models and current configuration.

    Returns:
        dict: Status information including:
            - default_provider: Currently configured default provider
            - ollama_available: Whether Ollama is accessible
            - openai_configured: Whether OpenAI API key is set
            - supported_providers: List of supported model providers
    """
    global config
    if config is None:
        config = get_config()

    status = {
        "default_provider": config.vision_default_provider,
        "supported_providers": ["none", "ollama", "openai"],
        "openai_configured": bool(config.openai_api_key),
        "ollama_base_url": config.ollama_base_url,
        "ollama_model": config.ollama_model,
        "openai_model": config.openai_model if config.openai_api_key else "Not configured",
        "max_image_size_mb": config.vision_max_image_size_mb,
    }

    # Check if Ollama is accessible
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{config.ollama_base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as response:
                status["ollama_available"] = response.status == 200
    except Exception as e:
        status["ollama_available"] = False
        status["ollama_error"] = str(e)

    return status


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
