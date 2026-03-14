# Local Read MCP Server
# A Model Context Protocol server for document processing

from .converters import (
    DocumentConverterResult,
    HtmlConverter,
    DocxConverter,
    XlsxConverter,
    PptxConverter,
    PdfConverter,
    TextConverter,
    JsonConverter,
    YamlConverter,
    CsvConverter,
    ZipConverter,
    MarkItDownConverter,
    extract_pdf_images,
    convert_html_to_md,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MEDIA_EXTENSIONS,
    PaginationManager,
    generate_session_id,
    apply_content_limit,
    extract_sections_from_markdown,
    fix_latex_formulas,
    html_to_markdown_result,
)

__version__ = "0.1.0"
__all__ = [
    # Converters
    "DocumentConverterResult",
    "HtmlConverter",
    "DocxConverter",
    "XlsxConverter",
    "PptxConverter",
    "PdfConverter",
    "TextConverter",
    "JsonConverter",
    "YamlConverter",
    "CsvConverter",
    "ZipConverter",
    "MarkItDownConverter",
    "extract_pdf_images",
    "convert_html_to_md",

    # Constants
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
]
