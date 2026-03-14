# Local Read MCP Converters
# A collection of document converters for various file formats

from .base import (
    DocumentConverterResult,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MEDIA_EXTENSIONS,
    _CustomMarkdownify
)

from .utils import (
    PaginationManager,
    generate_session_id,
    apply_content_limit,
    extract_sections_from_markdown,
    fix_latex_formulas,
    html_to_markdown_result
)

from .html import HtmlConverter, convert_html_to_md
from .docx import DocxConverter
from .xlsx import XlsxConverter
from .pptx import PptxConverter
from .pdf import PdfConverter, extract_pdf_images
from .simple import TextConverter, JsonConverter, YamlConverter, CsvConverter, MarkItDownConverter
from .zip import ZipConverter

__all__ = [
    # Base classes and constants
    "DocumentConverterResult",
    "IMAGE_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "MEDIA_EXTENSIONS",

    # Utilities
    "PaginationManager",
    "generate_session_id",
    "apply_content_limit",
    "extract_sections_from_markdown",
    "fix_latex_formulas",
    "html_to_markdown_result",

    # Converters
    "HtmlConverter",
    "convert_html_to_md",
    "DocxConverter",
    "XlsxConverter",
    "PptxConverter",
    "PdfConverter",
    "extract_pdf_images",
    "TextConverter",
    "JsonConverter",
    "YamlConverter",
    "CsvConverter",
    "MarkItDownConverter",
    "ZipConverter",
]
