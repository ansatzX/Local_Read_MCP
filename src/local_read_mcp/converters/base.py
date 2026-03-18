import json
import logging
import os
import re
import shutil
import tempfile
import time
import traceback
import csv as csv_module
from typing import Any, Union, Optional, Dict, List
from urllib.parse import quote, unquote, urlparse, urlunparse

# Optional imports
try:
    import mammoth
except ImportError:
    mammoth = None

try:
    import markdownify
except ImportError:
    markdownify = None

try:
    import openpyxl
    from openpyxl.utils import get_column_letter
except ImportError:
    openpyxl = None
    get_column_letter = None

try:
    import pdfminer
    import pdfminer.high_level
except ImportError:
    pdfminer = None

try:
    import pptx
except ImportError:
    pptx = None

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import html
except ImportError:
    html = None

# Media file extension constants
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
AUDIO_EXTENSIONS = {"wav", "mp3", "m4a"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class DocumentConverterResult:
    """Document conversion result with enhanced metadata and structure.

    This class encapsulates the result of converting a document to text/markdown format,
    including optional metadata, sections, tables, images, and pagination information.

    Attributes:
        title: Document title (optional, may be None)
        text_content: Converted text/markdown content
        metadata: Additional metadata dict (file size, timestamps, etc.)
        sections: List of extracted sections with headings
        tables: List of extracted table information
        images: List of extracted image information (for PDFs with extract_images=True)
        pagination_info: Pagination details (page count, offsets, etc.)
        processing_time_ms: Processing time in milliseconds (optional)
        error: Error message if conversion failed (optional)
        rendered_pages: List of rendered page information (for PDF rendering)
        extracted_tables: List of extracted table data
        form_fields: List of form field data
        structure: Document structure information
        text_with_coords: List of text with coordinate information

    Example:
        >>> result = DocumentConverterResult(
        ...     title="My Document",
        ...     text_content="# Heading\n\nContent here",
        ...     metadata={"file_size": 12345}
        ... )
        >>> print(result.title)
        My Document
    """

    def __init__(
        self,
        title: Union[str, None] = None,
        text_content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        pagination_info: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[int] = None,
        error: Optional[str] = None,
        # New fields
        rendered_pages: Optional[List[Dict[str, Any]]] = None,
        extracted_tables: Optional[List[Dict[str, Any]]] = None,
        form_fields: Optional[List[Dict[str, Any]]] = None,
        structure: Optional[Dict[str, Any]] = None,
        text_with_coords: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize document converter result.

        Args:
            title: Document title (optional)
            text_content: Converted text content
            metadata: Additional metadata (default: empty dict)
            sections: List of document sections (default: empty list)
            tables: List of extracted tables (default: empty list)
            images: List of extracted images (default: empty list)
            pagination_info: Pagination information (default: empty dict)
            processing_time_ms: Processing time in milliseconds (optional)
            error: Error message if failed (optional)
            rendered_pages: List of rendered page information (default: empty list)
            extracted_tables: List of extracted table data (default: empty list)
            form_fields: List of form field data (default: empty list)
            structure: Document structure information (default: empty dict)
            text_with_coords: List of text with coordinate information (default: empty list)
        """
        self.title: Union[str, None] = title
        self.text_content: str = text_content
        self.metadata = metadata or {}
        self.sections = sections or []
        self.tables = tables or []
        self.images = images or []
        self.pagination_info = pagination_info or {}
        self.processing_time_ms = processing_time_ms
        self.error = error
        self.rendered_pages = rendered_pages or []
        self.extracted_tables = extracted_tables or []
        self.form_fields = form_fields or []
        self.structure = structure or {}
        self.text_with_coords = text_with_coords or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the conversion result.

        Example:
            >>> result = DocumentConverterResult(title="Test", text_content="Content")
            >>> d = result.to_dict()
            >>> d["title"]
            'Test'
        """
        result = {
            "title": self.title,
            "text_content": self.text_content,
            "metadata": self.metadata,
            "sections": self.sections,
            "tables": self.tables,  # Keep for backward compatibility
            "pagination_info": self.pagination_info,
        }
        # Add new fields only if they exist and are non-empty
        if hasattr(self, 'rendered_pages') and self.rendered_pages:
            result["rendered_pages"] = self.rendered_pages
        if hasattr(self, 'extracted_tables') and self.extracted_tables:
            result["extracted_tables"] = self.extracted_tables
        if hasattr(self, 'form_fields') and self.form_fields:
            result["form_fields"] = self.form_fields
        if hasattr(self, 'structure') and self.structure:
            result["structure"] = self.structure
        if hasattr(self, 'text_with_coords') and self.text_with_coords:
            result["text_with_coords"] = self.text_with_coords
        if self.processing_time_ms is not None:
            result["processing_time_ms"] = self.processing_time_ms
        if self.error is not None:
            result["error"] = self.error
        return result


# Conditionally define _CustomMarkdownify only if markdownify is available
_CustomMarkdownify = None
if markdownify is not None:
    class _CustomMarkdownify(markdownify.MarkdownConverter):
        """
        A custom version of markdownify's MarkdownConverter. Changes include:

        - Altering the default heading style to use '#', '##', etc.
        - Removing javascript hyperlinks.
        - Truncating images with large data:uri sources.
        - Ensuring URIs are properly escaped, and do not conflict with Markdown syntax
        """

        def __init__(self, **options: Any):
            options["heading_style"] = options.get("heading_style", markdownify.ATX)
            # Explicitly cast options to the expected type if necessary
            super().__init__(**options)

        def convert_hn(self, n: int, el: Any, text: str, convert_as_inline: bool) -> str:
            """Same as usual, but be sure to start with a new line"""
            if not convert_as_inline:
                if not re.search(r"^\n", text):
                    return "\n" + super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

            return super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

        def convert_a(self, el: Any, text: str, convert_as_inline: bool):
            """Same as usual converter, but removes Javascript links and escapes URIs."""
            prefix, suffix, text = markdownify.chomp(text)  # type: ignore
            if not text:
                return ""
            href = el.get("href")
            title = el.get("title")

            # Escape URIs and skip non-http or file schemes
            if href:
                try:
                    parsed_url = urlparse(href)  # type: ignore
                    if parsed_url.scheme and parsed_url.scheme.lower() not in [
                        "http",
                        "https",
                        "file",
                    ]:  # type: ignore
                        return "%s%s%s" % (prefix, text, suffix)
                    href = urlunparse(
                        parsed_url._replace(path=quote(unquote(parsed_url.path)))
                    )  # type: ignore
                except ValueError:  # It's not clear if this ever gets thrown
                    return "%s%s%s" % (prefix, text, suffix)

            # For the replacement see #29: text nodes underscores are escaped
            if (
                self.options["autolinks"]
                and text.replace(r"\_", "_") == href
                and not title
                and not self.options["default_title"]
            ):
                # Shortcut syntax
                return "<%s>" % href
            if self.options["default_title"] and not title:
                title = href
            title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
            return (
                "%s[%s](%s%s)%s" % (prefix, text, href, title_part, suffix)
                if href
                else text
            )

        def convert_img(self, el: Any, text: str, convert_as_inline: bool) -> str:
            """Same as usual converter, but removes data URIs"""

            alt = el.attrs.get("alt", None) or ""
            src = el.attrs.get("src", None) or ""
            title = el.attrs.get("title", None) or ""
            title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
            if (
                convert_as_inline
                and el.parent.name not in self.options["keep_inline_images_in"]
            ):
                return alt

            # Remove dataURIs
            if src.startswith("data:"):
                src = src.split(",")[0] + "..."

            return "![%s](%s%s)" % (alt, src, title_part)

        def convert_soup(self, soup: Any) -> str:
            return super().convert_soup(soup)  # type: ignore
