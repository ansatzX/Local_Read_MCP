"""Vision-related functionality for the Local Read MCP Server."""

import base64
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def guess_mime_type_from_extension(file_path: str) -> str:
    """Guess MIME type based on file extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    mime_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    return mime_types.get(ext, "image/jpeg")


async def call_vision_api(
    image_path: str,
    question: str,
    api_key: str,
    base_url: str,
    model: str
) -> str:
    """Call OpenAI-compatible vision API (Doubao, GPT-4o, etc.)."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return "Error: openai package not installed. Install with: pip install openai"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    mime_type = guess_mime_type_from_extension(image_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
        ]
    }]

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Vision API error: {e}")
        return f"Error: Vision API call failed: {str(e)}"
