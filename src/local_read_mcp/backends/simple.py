# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
Simple backend implementation using existing converters.
"""

from pathlib import Path
from typing import Dict, Any
import logging

from .base import DocumentBackend
from ..intermediate_json import IntermediateJSONBuilder
from ..converters import (
    PdfConverter,
    DocxConverter,
    XlsxConverter,
    PptxConverter,
    HtmlConverter,
    ZipConverter,
    TextConverter,
    JsonConverter,
    YamlConverter,
    CsvConverter,
    MarkItDownConverter
)

logger = logging.getLogger(__name__)


class SimpleBackend(DocumentBackend):
    """Simple backend using existing converters."""

    @property
    def name(self) -> str:
        return "Simple"

    @property
    def description(self) -> str:
        return "Simple backend using existing converters (no ML models required)"

    @property
    def available(self) -> bool:
        return True

    @property
    def warning(self) -> None:
        return None

    def process(
        self,
        file_path: Path,
        format: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a document using simple converters."""
        logger.info(f"Processing with Simple backend: {file_path}")

        # Get file size
        file_size = file_path.stat().st_size if file_path.exists() else None

        # Create builder
        builder = IntermediateJSONBuilder(
            source_path=str(file_path.absolute()),
            source_format=format,
            page_count=1,  # Simple backend doesn't do page counting
            file_size=file_size
        )

        # Get converter and convert
        try:
            converter = self._get_converter(format)
            if converter:
                # Pass kwargs to converter (converter is a function)
                result = converter(str(file_path), **kwargs)
                markdown_content = result.text_content
                # Handle images if they were extracted
                if hasattr(result, 'images') and result.images:
                    logger.info(f"Extracted {len(result.images)} images")
            else:
                # Fallback to MarkItDownConverter
                markdown_content = MarkItDownConverter(str(file_path)).text_content
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            markdown_content = f"Error converting file: {e}"

        # Add as a single text block
        builder.add_block(
            type="text",
            page=1,
            bbox=[0, 0, 612, 792],  # Default page size
            content=markdown_content
        )

        return builder.build()

    def _get_converter(self, format: str):
        """Get the appropriate converter for the format."""
        converter_map = {
            "pdf": PdfConverter,
            "word": DocxConverter,
            "excel": XlsxConverter,
            "ppt": PptxConverter,
            "html": HtmlConverter,
            "zip": ZipConverter,
            "text": TextConverter,
            "json": JsonConverter,
            "yaml": YamlConverter,
            "csv": CsvConverter,
        }
        return converter_map.get(format)
