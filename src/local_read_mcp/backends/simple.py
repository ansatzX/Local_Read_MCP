# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
Simple backend implementation using existing converters.
"""

import logging
from pathlib import Path
from typing import Any

from ..converters import (
    CsvConverter,
    DocxConverter,
    HtmlConverter,
    JsonConverter,
    MarkItDownConverter,
    PdfConverter,
    PptxConverter,
    TextConverter,
    XlsxConverter,
    YamlConverter,
    ZipConverter,
)
from ..intermediate_json import IntermediateJSONBuilder
from .base import DocumentBackend

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
    ) -> dict[str, Any]:
        """Process a document using simple converters."""
        logger.info(f"Processing with Simple backend: {file_path}")

        # Get file size
        try:
            file_size = file_path.stat().st_size
        except (FileNotFoundError, OSError):
            file_size = None

        # Get converter and convert
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
            result = None

        # Get page count from result if available
        page_count = 1
        if result and hasattr(result, 'pagination_info'):
            page_count = result.pagination_info.get('page_count', 1)

        # Create builder
        builder = IntermediateJSONBuilder(
            source_path=str(file_path.absolute()),
            source_format=format,
            page_count=page_count,
            file_size=file_size
        )

        # Set metadata if available
        if result:
            metadata = {}
            if result.title:
                metadata['title'] = result.title
            if hasattr(result, 'metadata') and result.metadata:
                # Map common metadata fields
                if 'author' in result.metadata:
                    metadata['author'] = str(result.metadata['author'])
                if 'subject' in result.metadata:
                    metadata['subject'] = str(result.metadata['subject'])
                if 'keywords' in result.metadata:
                    metadata['keywords'] = str(result.metadata['keywords'])
                if 'created_at' in result.metadata:
                    metadata['created_at'] = str(result.metadata['created_at'])
                if 'modified_at' in result.metadata:
                    metadata['modified_at'] = str(result.metadata['modified_at'])
            if metadata:
                builder.set_metadata(**metadata)

        # Add structured blocks if result has them
        current_page = 1
        bbox = [0, 0, 612, 792]  # Default page size

        # Add sections as section blocks
        if result and hasattr(result, 'sections') and result.sections:
            for section in result.sections:
                section_type = section.get('type', 'section')
                level = section.get('level', 1)
                title = section.get('title', '')
                content = section.get('content', '')

                if title:
                    builder.add_block(
                        type=section_type,
                        page=current_page,
                        bbox=bbox,
                        level=level,
                        title=title,
                        content=content
                    )

        # Add tables as table blocks
        if result and hasattr(result, 'tables') and result.tables:
            for table in result.tables:
                table_content = table.get('markdown', '') or table.get('content', '')
                if table_content:
                    builder.add_block(
                        type='table',
                        page=current_page,
                        bbox=bbox,
                        content=table_content
                    )

        # Add images as image blocks
        if result and hasattr(result, 'images') and result.images:
            for image in result.images:
                image_path = image.get('path', '')
                description = image.get('description', '')
                builder.add_block(
                    type='image',
                    page=current_page,
                    bbox=bbox,
                    path=image_path,
                    description=description
                )

        # Always add the main text content as a fallback
        if markdown_content:
            builder.add_block(
                type="text",
                page=current_page,
                bbox=bbox,
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
