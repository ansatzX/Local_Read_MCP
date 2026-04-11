# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
VLM backends for document parsing using Vision Language Models.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any

from ..config import get_config
from ..intermediate_json import IntermediateJSONBuilder
from .base import DocumentBackend
from .model_detector import get_model_detector

logger = logging.getLogger(__name__)


class OpenAIVLBackend(DocumentBackend):
    """OpenAI VLM backend for document parsing using GPT-4V or similar models."""

    def __init__(self):
        self._detector = get_model_detector()
        self._config = get_config()

    @property
    def name(self) -> str:
        return "OpenAI VLM"

    @property
    def backend_identifier(self) -> str:
        """Backend identifier used in placeholder content (can be overridden)."""
        return "OpenAI VLM backend"

    @property
    def description(self) -> str:
        return "Document parsing using OpenAI Vision Language Models (GPT-4V, etc.)"

    @property
    def available(self) -> bool:
        """Check if OpenAI VLM backend is available."""
        return self._detector.vlm_available

    @property
    def warning(self) -> str | None:
        """Get warning message if OpenAI VLM backend is not available."""
        if self.available:
            return None
        return self._detector.vlm_warning

    def supports_format(self, format: str) -> bool:
        """Check if VLM backend supports the given format."""
        supported_formats = ["pdf", "png", "jpg", "jpeg", "gif", "webp"]
        return format in supported_formats

    def _get_file_size(self, file_path: Path) -> int | None:
        """Get file size safely."""
        try:
            return file_path.stat().st_size
        except FileNotFoundError:
            return None

    def _encode_image(self, image_path: Path) -> str:
        """Encode an image file to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _call_openai_vlm(self, images: list, prompt: str) -> str:
        """Call OpenAI VLM API with images and prompt."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("OpenAI Python package not installed")

        client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url
        )

        # Prepare messages content
        content = [{"type": "text", "text": prompt}]

        for image_data in images:
            if isinstance(image_data, Path):
                # Encode image file
                base64_image = self._encode_image(image_data)
                image_url = f"data:image/jpeg;base64,{base64_image}"
            elif isinstance(image_data, str) and image_data.startswith("data:"):
                # Already a data URL
                image_url = image_data
            else:
                # Assume it's a file path
                base64_image = self._encode_image(Path(image_data))
                image_url = f"data:image/jpeg;base64,{base64_image}"

            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        # Call API
        response = client.chat.completions.create(
            model=self._config.model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=4096
        )

        return response.choices[0].message.content

    def _convert_vlm_response_to_intermediate(
        self,
        vlm_response: str,
        file_path: Path,
        format: str,
        file_size: int | None = None
    ) -> dict[str, Any]:
        """Convert VLM response to our intermediate JSON format."""
        # Get file size if not provided
        if file_size is None:
            file_size = self._get_file_size(file_path)

        # Create builder
        builder = IntermediateJSONBuilder(
            source_path=str(file_path.absolute()),
            source_format=format,
            page_count=1,
            file_size=file_size
        )

        # Try to parse JSON response if it's JSON
        content = vlm_response
        try:
            parsed = json.loads(vlm_response)
            if isinstance(parsed, dict):
                if "content" in parsed:
                    content = str(parsed["content"])
                elif "markdown" in parsed:
                    content = str(parsed["markdown"])
                elif "text" in parsed:
                    content = str(parsed["text"])
        except json.JSONDecodeError:
            # Not JSON, use as is
            pass

        # Add as a single text block
        builder.add_block(
            type="text",
            page=1,
            bbox=[0, 0, 612, 792],
            content=content
        )

        return builder.build()

    def process(
        self,
        file_path: Path,
        format: str,
        **kwargs
    ) -> dict[str, Any]:
        """Process a document using OpenAI VLM backend."""
        logger.info(f"Processing with {self.name} backend: {file_path}")

        if not self.available:
            raise RuntimeError(f"{self.name} backend is not available")

        # Check if format is supported
        if not self.supports_format(format):
            raise ValueError(f"{self.name} backend does not support format: {format}")

        # Calculate file size once
        file_size = self._get_file_size(file_path)

        # Get prompt or use default
        prompt = kwargs.get("prompt", self._get_default_prompt(format))

        # Try to process the document
        images = []
        if format == "pdf":
            # Render PDF to images
            images = self._render_pdf_to_images(file_path, kwargs.get('max_pages', 3))
        elif format in ["png", "jpg", "jpeg", "gif", "webp"]:
            # Image file, use directly
            images = [file_path]

        if not images:
            raise RuntimeError(f"Could not process {format} file: no images could be extracted/rendered")

        # Call VLM API with images
        vlm_response = self._call_openai_vlm(images, prompt)
        return self._convert_vlm_response_to_intermediate(
            vlm_response, file_path, format, file_size
        )

    def _render_pdf_to_images(self, pdf_path: Path, max_pages: int = 3) -> list:
        """Render PDF pages to images."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed, cannot render PDF")
            return []

        images = []
        try:
            doc = fitz.open(pdf_path)
            # Render first N pages
            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                # Render page to image
                pix = page.get_pixmap(dpi=150)
                # Convert to PIL Image
                from PIL import Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Save to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img.save(tmp, format="PNG")
                    images.append(Path(tmp.name))

            doc.close()
        except Exception as e:
            logger.warning(f"Failed to render PDF: {e}")

        return images

    def _get_default_prompt(self, format: str) -> str:
        """Get default prompt for document parsing."""
        return f"""Please analyze this {format} document and extract its content.

Provide the output as a JSON object with the following structure:
{{
    "title": "Document title (if any)",
    "content": "Full document content in Markdown format",
    "sections": [
        {{"title": "Section title", "level": 1}}
    ],
    "tables": [
        {{"description": "Table description", "markdown": "Table in Markdown format"}}
    ],
    "images": [
        {{"description": "Image description"}}
    ]
}}

If you cannot output JSON, provide the content in Markdown format."""


class QwenVLBackend(OpenAIVLBackend):
    """Qwen-VL backend for document parsing using Qwen's Vision Language Models."""

    @property
    def name(self) -> str:
        return "Qwen-VL"

    @property
    def backend_identifier(self) -> str:
        """Backend identifier used in placeholder content."""
        return "Qwen-VL backend"

    @property
    def description(self) -> str:
        return "Document parsing using Qwen Vision Language Models"
