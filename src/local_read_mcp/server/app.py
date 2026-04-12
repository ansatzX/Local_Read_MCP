# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
Local Read MCP Server

A Model Context Protocol server for processing various file formats.
Converts documents to markdown/text without requiring external APIs.
"""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..backends import BackendType, get_registry
from ..config import get_config as _get_config
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
    extract_sections_from_markdown,
    generate_session_id,
)
from ..index_generator import IndexGenerator
from ..markdown_converter import MarkdownConverter
from ..output_manager import OutputManager
from .utils import (
    apply_pagination,
    create_simple_converter_wrapper,
    duplicate_detector,
    fix_tool_arguments,
)
from .vision import call_vision_api

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("local_read_mcp-server")

# Initialize config to check if vision features should be enabled
_config = _get_config()
VISION_ENABLED = _config.vision_enabled

if VISION_ENABLED:
    logger.info(f"Vision features ENABLED (model: {_config.model})")
else:
    logger.info("Vision features DISABLED - configure VISION_API_KEY or OPENAI_API_KEY in .env file to enable")


_SIMPLE_CONVERTER_BUILDERS: dict[str, tuple[Callable[..., Any], str]] = {
    "text": (TextConverter, "text"),
    "json": (JsonConverter, "json"),
    "csv": (CsvConverter, "csv"),
    "yaml": (YamlConverter, "yaml"),
    "ppt": (PptxConverter, "pptx"),
    "zip": (ZipConverter, "zip"),
    "markitdown": (MarkItDownConverter, "markitdown"),
}
_SIMPLE_CONVERTER_CACHE: dict[str, Callable[..., Any]] = {}

_FORMAT_BY_EXTENSION: dict[str, str] = {
    # Text formats
    ".txt": "text",
    ".md": "text",
    ".py": "text",
    ".sh": "text",
    ".log": "text",
    ".rst": "text",
    ".json": "json",
    ".csv": "csv",
    ".yaml": "yaml",
    ".yml": "yaml",
    # Binary/document formats
    ".pdf": "pdf",
    ".docx": "word",
    ".doc": "word",
    ".xlsx": "excel",
    ".xls": "excel",
    ".pptx": "ppt",
    ".ppt": "ppt",
    ".html": "html",
    ".htm": "html",
    ".zip": "zip",
}

_TEXT_FORMAT_GROUPS: list[tuple[str, str]] = [
    ("Plain Text", "text"),
    ("JSON", "json"),
    ("CSV", "csv"),
    ("YAML", "yaml"),
]
_BINARY_FORMAT_GROUPS: list[tuple[str, str]] = [
    ("PDF", "pdf"),
    ("Word", "word"),
    ("Excel", "excel"),
    ("PowerPoint", "ppt"),
    ("HTML", "html"),
    ("ZIP", "zip"),
]
def get_simple_converter_wrapper(format_name: str) -> Callable[..., Any]:
    """Get (and cache) wrapper for simple converters."""
    if format_name in _SIMPLE_CONVERTER_CACHE:
        return _SIMPLE_CONVERTER_CACHE[format_name]

    converter_func, converter_name = _SIMPLE_CONVERTER_BUILDERS[format_name]
    wrapper = create_simple_converter_wrapper(converter_func, converter_name)
    _SIMPLE_CONVERTER_CACHE[format_name] = wrapper
    return wrapper


def _extract_common_read_params(
    tool_name: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract shared read_* parameters after applying argument auto-fixes."""
    fixed_params = fix_tool_arguments(tool_name, params)
    common_params = {
        "file_path": fixed_params.get("file_path", params.get("file_path")),
        "format": fixed_params.get("format", params.get("format")),
        "chunk": fixed_params.get("chunk", params.get("chunk")),
        "chunk_size": fixed_params.get("chunk_size", params.get("chunk_size")),
        "offset": fixed_params.get("offset", params.get("offset")),
        "limit": fixed_params.get("limit", params.get("limit")),
        "extract_sections": fixed_params.get("extract_sections", params.get("extract_sections")),
        "extract_tables": fixed_params.get("extract_tables", params.get("extract_tables")),
        "extract_metadata": fixed_params.get("extract_metadata", params.get("extract_metadata")),
        "preview_only": fixed_params.get("preview_only", params.get("preview_only")),
        "preview_lines": fixed_params.get("preview_lines", params.get("preview_lines")),
        "session_id": fixed_params.get("session_id", params.get("session_id")),
        "return_format": fixed_params.get("return_format", params.get("return_format")),
    }
    return common_params, fixed_params


def _extensions_for_format(format_key: str) -> list[str]:
    """Return all extensions mapped to a specific format key."""
    return [ext for ext, mapped_format in _FORMAT_BY_EXTENSION.items() if mapped_format == format_key]


def _build_supported_format_groups() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build supported format groups from shared extension mapping constants."""
    text_formats = [
        {"name": display_name, "extensions": _extensions_for_format(format_key)}
        for display_name, format_key in _TEXT_FORMAT_GROUPS
    ]
    binary_formats = [
        {"name": display_name, "extensions": _extensions_for_format(format_key)}
        for display_name, format_key in _BINARY_FORMAT_GROUPS
    ]
    return text_formats, binary_formats


def detect_format(file_path: str) -> str | None:
    """Detect file format from extension.

    Returns:
        Format string or None for unknown (use markitdown fallback)
    """
    ext = os.path.splitext(file_path)[1].lower()

    return _FORMAT_BY_EXTENSION.get(ext)


async def process_pdf_document(
    file_path: str,
    chunk: int | None,
    chunk_size: int | None,
    offset: int | None,
    limit: int | None,
    extract_sections: bool | None,
    extract_tables: bool | None,
    extract_metadata: bool | None,
    extract_images: bool | None,
    render_images: bool | None,
    render_dpi: int | None,
    render_format: str | None,
    extract_forms: bool | None,
    inspect_struct: bool | None,
    include_coords: bool | None,
    images_output_dir: str | None,
    preview_only: bool | None,
    preview_lines: int | None,
    session_id: str | None,
    return_format: str | None,
) -> dict[str, Any]:
    """Process PDF document with enhanced features.

    Handles PDF-specific features: text rendering, images, tables, forms, structure inspection.
    """
    start_time = time.time()

    try:
        # Get full content from converter (always extract metadata to get PDF page count)
        result = PdfConverter(
            file_path,
            extract_metadata=True,
            extract_images=extract_images,
            images_output_dir=images_output_dir,
            render_images=render_images,
            render_dpi=render_dpi,
            render_format=render_format,
            extract_tables=extract_tables,
            extract_forms=extract_forms,
            inspect_struct=inspect_struct,
            include_coords=include_coords,
        )
        full_content = result.text_content

        # Get PDF page count from metadata
        pdf_page_count = result.metadata.get("pdf_page_count") if result.metadata else None

        # Calculate pagination parameters
        if offset is not None:
            char_offset = offset
            char_limit = limit
        else:
            if chunk is None or chunk < 1:
                chunk = 1
            if chunk_size is None or chunk_size < 1:
                chunk_size = 10000
            char_offset = (chunk - 1) * chunk_size
            char_limit = chunk_size

        # Apply pagination
        paginated_content, has_more = apply_pagination(full_content, char_offset, char_limit)

        # Apply preview if requested (does not affect has_more flag)
        if preview_only:
            lines = paginated_content.split('\n')
            total_lines = len(lines)
            if total_lines > preview_lines:
                paginated_content = '\n'.join(lines[:preview_lines]) + f"\n\n... [Preview mode: showing first {preview_lines} of {total_lines} lines. Remove preview_only to read full content]"

        # Extract structured data if requested
        metadata = {}
        sections = []
        tables = []
        images = result.images if hasattr(result, 'images') else []

        if extract_metadata:
            file_size = None
            try:
                file_size = os.path.getsize(file_path)
            except (OSError, Exception):
                pass
            metadata = {
                "title": result.title,
                "file_path": file_path,
                "file_size": file_size,
                "pdf_page_count": pdf_page_count,
            }
            # Add image metadata if images were extracted
            if extract_images and images:
                metadata["image_count"] = len(images)
                metadata["images_directory"] = images_output_dir or result.metadata.get("images_directory")

        if extract_sections:
            sections = extract_sections_from_markdown(full_content)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Generate session ID if not provided
        if not session_id:
            session_id = generate_session_id(file_path, prefix="pdf")

        # Prepare result based on return format
        if return_format.lower() == "json":
            result_dict = {
                "success": True,
                "text": paginated_content,
                "content": paginated_content,  # Keep for compatibility
                "title": result.title,
                "metadata": metadata,
                "sections": sections,
                "tables": tables,
                "images": images if extract_images else [],
                "pagination_info": {
                    "current_chunk": chunk if offset is None else None,
                    "total_chunks": max(1, (len(full_content) + char_limit - 1) // char_limit) if char_limit else 1,
                    "chunk_size": char_limit,
                    "has_more": has_more,
                    "char_offset": char_offset,
                    "char_limit": char_limit,
                    "total_chars": len(full_content),
                },
                "pdf_pages": pdf_page_count,  # Actual PDF page count (separate from chunk pagination)
                "session_id": session_id,
                "processing_time_ms": processing_time_ms,
            }

            # Include new fields if present
            if hasattr(result, 'rendered_pages') and result.rendered_pages:
                result_dict["rendered_pages"] = result.rendered_pages
            if hasattr(result, 'extracted_tables') and result.extracted_tables:
                result_dict["extracted_tables"] = result.extracted_tables
            if hasattr(result, 'form_fields') and result.form_fields:
                result_dict["form_fields"] = result.form_fields
            if hasattr(result, 'structure') and result.structure:
                result_dict["structure"] = result.structure
            if hasattr(result, 'text_with_coords') and result.text_with_coords:
                result_dict["text_with_coords"] = result.text_with_coords

            # Check if response might be too large for token limits
            if len(sections) > 30 or (len(paginated_content) > 8000 and len(sections) > 0):
                result_dict["warning"] = (
                    f"Large response with {len(sections)} sections. "
                    f"If you encounter token limit errors, try using smaller chunk_size (5000-8000) "
                    f"or switch to return_format='text'."
                )

            return result_dict
        else:
            # Return text format with pagination hints
            result_text = paginated_content

            # Add image extraction info if requested
            if extract_images and images:
                images_info = f"\n\n[Extracted {len(images)} images from PDF. Use return_format='json' to see image details]"
                result_text = images_info + "\n" + result_text

            # Add pagination hint if there's more content
            if has_more:
                total_chars = len(full_content)
                current_end = char_offset + len(paginated_content)
                total_chunks = max(1, (total_chars + char_limit - 1) // char_limit)
                pdf_info = f" | PDF: {pdf_page_count} pages" if pdf_page_count else ""
                result_text += f"\n\n[Chunk {chunk}/{total_chunks}: Characters {char_offset:,}-{current_end:,} of {total_chars:,}{pdf_info}. Continue with chunk={chunk + 1}]"

            return {
                "success": True,
                "text": result_text,
                "content": result_text,
                "title": result.title,
                "has_more": has_more,
                "total_chars": len(full_content),
                "current_chunk": chunk if offset is None else None,
                "pdf_pages": pdf_page_count,
                "image_count": len(images) if extract_images else 0,
            }

    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        processing_time_ms = int((time.time() - start_time) * 1000)

        if return_format.lower() == "json":
            return {
                "success": False,
                "error": str(e),
                "content": f"Error: Failed to read PDF file: {e!s}",
                "processing_time_ms": processing_time_ms,
            }
        else:
            return {
                "success": False,
                "error": str(e),
                "content": f"Error: Failed to read PDF file: {e!s}",
            }


@mcp.tool()
async def analyze_image(
    image_path: str,
    question: str = "Describe this image in detail. What type of content is it?",
    api_key: str | None = None
) -> str:
    """Analyze an image using OpenAI-compatible vision API.

    Args:
        image_path: Path to the image file to analyze
        question: Question to ask about the image
        api_key: API key (overrides config if provided)

    Returns:
        Answer from the vision model

    Environment Variables (.env):
        VISION_API_KEY: Your API key (or OPENAI_API_KEY)
        VISION_BASE_URL: API base URL (or OPENAI_BASE_URL)
        VISION_MODEL: Model name (or OPENAI_VISION_MODEL, default: gpt-4o)
        VISION_MAX_IMAGE_SIZE_MB: Max image size in MB (default: 20)
    """
    if not VISION_ENABLED:
        return "Vision is not enabled. Set VISION_API_KEY or OPENAI_API_KEY in .env file."

    if not os.path.exists(image_path):
        return f"Error: Image file not found: {image_path}"

    # Check file size
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    max_size = _config.vision_max_image_size_mb
    if file_size_mb > max_size:
        return f"Error: Image too large ({file_size_mb:.2f}MB). Maximum: {max_size}MB"

    # Use provided API key or config
    effective_api_key = api_key or _config.api_key
    if not effective_api_key:
        return "Error: API key not configured. Set VISION_API_KEY or OPENAI_API_KEY in .env."

    return await call_vision_api(
        image_path=image_path,
        question=question,
        api_key=effective_api_key,
        base_url=_config.base_url,
        model=_config.model
    )


@mcp.tool()
async def get_vision_status() -> dict[str, Any]:
    """Get vision server status and configuration.

    Returns:
        Status information about vision API configuration
    """
    return {
        "vision_enabled": VISION_ENABLED,
        "message": "Vision features available" if VISION_ENABLED else "Vision features not configured",
        "suggestion": (
            "Configure VISION_API_KEY and VISION_BASE_URL in .env file to enable vision features."
            if not VISION_ENABLED else None
        ),
        "configured": {
            "base_url": _config.base_url,
            "model": _config.model,
            "has_api_key": bool(_config.api_key),
            "max_image_size_mb": _config.vision_max_image_size_mb,
        } if VISION_ENABLED else None,
    }


async def process_document(
    file_path: str,
    converter_func,
    converter_kwargs: dict[str, Any],
    chunk: int | None = 1,
    chunk_size: int | None = 10000,
    offset: int | None = None,
    limit: int | None = None,
    extract_sections: bool | None = False,
    extract_tables: bool | None = False,
    extract_metadata: bool | None = False,
    preview_only: bool | None = False,
    preview_lines: int | None = 100,
    session_id: str | None = None,
    return_format: str | None = "text"
) -> dict[str, Any]:
    """
    Unified document processing with chunked pagination, session management, and structured extraction.

    Args:
        file_path: Path to the file
        converter_func: Converter function to use
        converter_kwargs: Keyword arguments to pass to converter function
        chunk: Chunk number for pagination (1-indexed)
        chunk_size: Number of characters per chunk
        offset: Character offset (alternative to chunk)
        limit: Character limit (alternative to chunk_size)
        extract_sections: Whether to extract document sections/headings
        extract_tables: Whether to extract tables
        extract_metadata: Whether to extract metadata
        preview_only: Whether to return only a preview
        preview_lines: Number of lines for preview mode
        session_id: Session ID for resuming pagination
        return_format: Output format: 'json' (structured) or 'text' (plain)

    Returns:
        Dictionary with content and metadata
    """
    start_time = time.time()

    # Generate session ID early for duplicate detection
    if not session_id:
        session_id = generate_session_id(file_path, prefix=converter_func.__name__.lower().replace('converter', ''))

    # Normalize chunk parameters (handle None and invalid values)
    if chunk is None or chunk < 1:
        chunk = 1
    if chunk_size is None or chunk_size < 1:
        chunk_size = 10000

    # Check for duplicate requests (prevent infinite loops)
    duplicate_warning = None
    if not preview_only:  # Skip detection for preview requests
        duplicate_warning = duplicate_detector.check_and_record(
            session_id=session_id,
            file_path=file_path,
            chunk=chunk,
            chunk_size=chunk_size
        )

    try:
        # Call converter function with appropriate kwargs
        result = converter_func(file_path, **converter_kwargs)

        if result.error:
            raise Exception(f"Converter error: {result.error}")

        full_content = result.text_content

        # Calculate pagination parameters
        if offset is not None:
            char_offset = offset
            char_limit = limit
        else:
            char_offset = (chunk - 1) * chunk_size
            char_limit = chunk_size

        # Apply pagination
        paginated_content, has_more = apply_pagination(full_content, char_offset, char_limit)

        # Apply preview if requested (does not affect has_more flag)
        if preview_only:
            lines = paginated_content.split('\n')
            total_lines = len(lines)
            if total_lines > preview_lines:
                paginated_content = '\n'.join(lines[:preview_lines]) + f"\n\n... [Preview mode: showing first {preview_lines} of {total_lines} lines. Remove preview_only to read full content]"

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Prepare result based on return format
        if return_format.lower() == "json":
            sections_list = result.sections if extract_sections else []
            tables_list = result.tables if extract_tables else []

            result_dict = {
                "success": True,
                "text": paginated_content,
                "content": paginated_content,  # Keep for compatibility
                "title": result.title,
                "metadata": {**result.metadata, **{
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                }} if extract_metadata else {},
                "sections": sections_list,
                "tables": tables_list,
                "pagination_info": {
                    "current_chunk": chunk if offset is None else None,
                    "total_chunks": max(1, (len(full_content) + char_limit - 1) // char_limit) if char_limit else 1,
                    "chunk_size": char_limit,
                    "has_more": has_more,
                    "char_offset": char_offset,
                    "char_limit": char_limit,
                    "total_chars": len(full_content),
                },
                "session_id": session_id,
                "processing_time_ms": processing_time_ms,
            }

            # Check if response might be too large for token limits
            if len(sections_list) > 30 or len(tables_list) > 20 or (len(paginated_content) > 8000 and (len(sections_list) > 0 or len(tables_list) > 0)):
                result_dict["warning"] = (
                    f"Large response with {len(sections_list)} sections and {len(tables_list)} tables. "
                    f"If you encounter token limit errors, try using smaller chunk_size (5000-8000) "
                    f"or switch to return_format='text'."
                )

            # Add duplicate warning if detected
            if duplicate_warning:
                # Combine with existing warning if present
                if "warning" in result_dict:
                    result_dict["warning"] = f"{result_dict['warning']}\n\n{duplicate_warning}"
                else:
                    result_dict["warning"] = duplicate_warning

            return result_dict
        else:
            # Return text format with pagination hints
            result_text = paginated_content

            # Add duplicate warning if detected
            if duplicate_warning:
                result_text = f"[{duplicate_warning}]\n\n{result_text}"

            # Add pagination hint if there's more content
            if has_more:
                total_chars = len(full_content)
                current_end = char_offset + len(paginated_content)
                total_chunks = max(1, (total_chars + char_limit - 1) // char_limit)
                result_text += f"\n\n[Chunk {chunk}/{total_chunks}: Characters {char_offset:,}-{current_end:,} of {total_chars:,}. Continue with chunk={chunk + 1}]"

            return {
                "success": True,
                "text": result_text,
                "content": result_text,
                "title": result.title,
                "has_more": has_more,
                "total_chars": len(full_content),
                "current_chunk": chunk if offset is None else None,
            }

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        processing_time_ms = int((time.time() - start_time) * 1000)

        if return_format.lower() == "json":
            return {
                "success": False,
                "error": str(e),
                "content": f"Error: Failed to process file: {e!s}",
                "processing_time_ms": processing_time_ms,
            }
        else:
            return {
                "success": False,
                "error": str(e),
                "content": f"Error: Failed to process file: {e!s}",
            }


@mcp.tool()
async def read_text_file(
    file_path: str,
    format: str | None = None,
    chunk: int | None = 1,
    chunk_size: int | None = 10000,
    offset: int | None = None,
    limit: int | None = None,
    extract_sections: bool | None = False,
    extract_tables: bool | None = False,
    extract_metadata: bool | None = False,
    preview_only: bool | None = False,
    preview_lines: int | None = 100,
    session_id: str | None = None,
    return_format: str | None = "text",
    # Backend selection
    backend: str = "auto",
) -> dict[str, Any]:
    """Read text-based files.

    Supported formats: .txt, .md, .py, .sh, .json, .csv, .yaml, .yml

    Args:
        file_path: Path to the file to read
        format: Explicit format override (text/json/csv/yaml)
        chunk: Chunk number for content pagination (1-indexed). Default: 1.
        chunk_size: Number of characters per chunk. Default: 10000.
        offset: Character offset (alternative to chunk). If specified, overrides chunk.
        limit: Character limit (alternative to chunk_size). If specified, overrides chunk_size.
        extract_sections: Extract document sections/headings. Use for structured documents. Default: False.
        extract_tables: Extract table information (CSV only). Default: False.
        extract_metadata: Extract file metadata. Use with return_format="json". Default: False.
        preview_only: Return only first N lines without full conversion. Use for quick assessment. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Reuse for consecutive chunk requests.
        return_format: Output format: 'json' (structured with metadata/sections) or 'text' (plain). Default: 'text'.
        backend: Backend to use (auto, simple). Default: auto.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, pagination_info, session_id.
    """
    common_params, _ = _extract_common_read_params("read_text_file", locals().copy())

    file_path = common_params["file_path"]
    format = common_params["format"]
    chunk = common_params["chunk"]
    chunk_size = common_params["chunk_size"]
    offset = common_params["offset"]
    limit = common_params["limit"]
    extract_sections = common_params["extract_sections"]
    extract_tables = common_params["extract_tables"]
    extract_metadata = common_params["extract_metadata"]
    preview_only = common_params["preview_only"]
    preview_lines = common_params["preview_lines"]
    session_id = common_params["session_id"]
    return_format = common_params["return_format"]

    # Auto-detect format if not provided
    if not format:
        format = detect_format(file_path)

    # Use backend path
    try:
        from ..backends import BackendType, get_registry
        from ..converters.base import DocumentConverterResult
        from ..index_generator import IndexGenerator
        from ..markdown_converter import MarkdownConverter

        registry = get_registry()

        # Parse backend type
        try:
            backend_type = BackendType(backend)
        except ValueError:
            backend_type = BackendType.AUTO

        # Select backend
        if backend_type == BackendType.AUTO:
            backend_instance = registry.select_best(format)
        else:
            backend_instance = registry.get(backend_type)

        if backend_instance is None:
            backend_instance = registry.get(BackendType.SIMPLE)

        # Validate that the selected backend supports the format
        if not backend_instance.supports_format(format):
            raise ValueError(f"Backend '{backend_instance.name}' does not support format '{format}'")

        warnings = []
        if backend_instance.warning:
            warnings.append(backend_instance.warning)

        # Process with backend
        file_path_obj = Path(file_path)
        intermediate = backend_instance.process(file_path_obj, format)

        # Convert intermediate to markdown
        markdown_converter = MarkdownConverter(intermediate)
        full_content = markdown_converter.convert()

        # Get sections and tables from index
        index_generator = IndexGenerator(intermediate)
        index_data = index_generator.generate()

        # Create a wrapper converter function for process_document
        def backend_converter(fp: str, **kwargs):
            return DocumentConverterResult(
                title=intermediate.get("metadata", {}).get("title"),
                text_content=full_content,
                metadata=intermediate.get("source", {}),
                sections=index_data.get("sections", []),
                tables=index_data.get("tables", [])
            )

        # Delegate to process_document
        result = await process_document(
            file_path=file_path,
            converter_func=backend_converter,
            converter_kwargs={},
            chunk=chunk,
            chunk_size=chunk_size,
            offset=offset,
            limit=limit,
            extract_sections=extract_sections,
            extract_tables=extract_tables,
            extract_metadata=extract_metadata,
            preview_only=preview_only,
            preview_lines=preview_lines,
            session_id=session_id,
            return_format=return_format
        )

        # Add backend info to the result
        if result.get("success"):
            result["backend_used"] = backend_instance.name
            if warnings:
                result["warnings"] = warnings

        return result

    except Exception as e:
        logger.warning(f"Backend processing failed: {e}, falling back to original logic")

    # Original logic as fallback
    # Map format to converter
    converter_kwargs = {
        "extract_metadata": extract_metadata,
        "extract_sections": extract_sections,
        "extract_tables": extract_tables,
    }

    if format in {"text", "json", "csv", "yaml"}:
        converter_func = get_simple_converter_wrapper(format)
    else:
        # Unknown format - use markitdown fallback
        converter_func = get_simple_converter_wrapper("markitdown")

    return await process_document(
        file_path=file_path,
        converter_func=converter_func,
        converter_kwargs=converter_kwargs,
        chunk=chunk,
        chunk_size=chunk_size,
        offset=offset,
        limit=limit,
        extract_sections=extract_sections,
        extract_tables=extract_tables,
        extract_metadata=extract_metadata,
        preview_only=preview_only,
        preview_lines=preview_lines,
        session_id=session_id,
        return_format=return_format
    )


@mcp.tool()
async def read_binary_file(
    file_path: str,
    format: str | None = None,
    # Standard pagination
    chunk: int | None = 1,
    chunk_size: int | None = 10000,
    offset: int | None = None,
    limit: int | None = None,
    # Standard structured extraction
    extract_sections: bool | None = False,
    extract_tables: bool | None = False,
    extract_metadata: bool | None = False,
    preview_only: bool | None = False,
    preview_lines: int | None = 100,
    session_id: str | None = None,
    return_format: str | None = "text",
    # PDF-specific features (only used when format=pdf or auto-detected as pdf)
    extract_images: bool | None = None,
    render_images: bool | None = False,
    render_dpi: int | None = 200,
    render_format: str | None = "png",
    extract_forms: bool | None = False,
    inspect_struct: bool | None = False,
    include_coords: bool | None = False,
    images_output_dir: str | None = None,
    # Backend selection
    backend: str = "auto",
) -> dict[str, Any]:
    """Read binary/document files.

    Supported formats: .pdf, .docx, .doc, .xlsx, .xls, .pptx, .ppt, .html, .htm, .zip

    PDF-specific features (only available for PDF files):
    - render_images: Render pages to images
    - extract_forms: Extract form fields
    - inspect_struct: Get complete PDF structure
    - include_coords: Include text coordinates

    Note: PDF-specific parameters are ignored for non-PDF files.

    Args:
        file_path: Path to the file to read
        format: Explicit format override (pdf/word/excel/ppt/html/zip)
        chunk: Chunk number for content pagination (1-indexed). Default: 1.
        chunk_size: Number of characters per chunk. Default: 10000.
        offset: Character offset (alternative to chunk). If specified, overrides chunk.
        limit: Character limit (alternative to chunk_size). If specified, overrides chunk_size.
        extract_sections: Extract document sections/headings. Use for structured documents. Default: False.
        extract_tables: Extract table information. Default: False.
        extract_metadata: Extract file metadata (size, path, timestamp, PDF pages). Use with return_format="json". Default: False.
        preview_only: Return only first N lines without full conversion. Use for quick assessment. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Reuse for consecutive chunk requests.
        return_format: Output format: 'json' (structured with metadata/sections) or 'text' (plain). Default: 'text'.
        extract_images: Extract images from PDF. If omitted, auto-enabled when vision is configured; otherwise disabled.
        render_images: Render PDF pages to images. Default: False.
        render_dpi: DPI for rendered images. Default: 200.
        render_format: Format for rendered images (png or jpeg). Default: png.
        extract_forms: Extract form fields from PDF. Default: False.
        inspect_struct: Get complete PDF structure (metadata, outline, fonts, etc.). Default: False.
        include_coords: Include text with bounding box coordinates. Default: False.
        images_output_dir: Directory to save extracted/rendered images. If None, uses temporary directory. Default: None.
        backend: Backend to use (auto, simple, mineru, qwen-vl, openai-vlm). Default: auto.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, pagination_info, pdf_pages, images, session_id.
    """
    common_params, fixed_params = _extract_common_read_params("read_binary_file", locals().copy())

    file_path = common_params["file_path"]
    format = common_params["format"]
    chunk = common_params["chunk"]
    chunk_size = common_params["chunk_size"]
    offset = common_params["offset"]
    limit = common_params["limit"]
    extract_sections = common_params["extract_sections"]
    extract_tables = common_params["extract_tables"]
    extract_metadata = common_params["extract_metadata"]
    preview_only = common_params["preview_only"]
    preview_lines = common_params["preview_lines"]
    session_id = common_params["session_id"]
    return_format = common_params["return_format"]

    extract_images = fixed_params.get("extract_images", extract_images)
    render_images = fixed_params.get("render_images", render_images)
    render_dpi = fixed_params.get("render_dpi", render_dpi)
    render_format = fixed_params.get("render_format", render_format)
    extract_forms = fixed_params.get("extract_forms", extract_forms)
    inspect_struct = fixed_params.get("inspect_struct", inspect_struct)
    include_coords = fixed_params.get("include_coords", include_coords)
    images_output_dir = fixed_params.get("images_output_dir", images_output_dir)

    # Auto-enable extract_images if vision is configured and value was not explicitly set
    if extract_images is None:
        extract_images = bool(VISION_ENABLED)
        if extract_images:
            logger.info("Vision enabled: auto-enabling extract_images=True for PDF")

    # Auto-detect format if not provided
    if not format:
        format = detect_format(file_path)

    # Use backend path (default)
    try:
        from ..backends import BackendType, get_registry
        from ..converters.base import DocumentConverterResult
        from ..index_generator import IndexGenerator
        from ..markdown_converter import MarkdownConverter

        registry = get_registry()

        # Parse backend type
        try:
            backend_type = BackendType(backend)
        except ValueError:
            backend_type = BackendType.AUTO

        # Select backend
        if backend_type == BackendType.AUTO:
            backend_instance = registry.select_best(format)
        else:
            backend_instance = registry.get(backend_type)

        if backend_instance is None:
            backend_instance = registry.get(BackendType.SIMPLE)

        # Validate that the selected backend supports the format
        if not backend_instance.supports_format(format):
            raise ValueError(f"Backend '{backend_instance.name}' does not support format '{format}'")

        warnings = []
        if backend_instance.warning:
            warnings.append(backend_instance.warning)

        # Process with backend
        file_path_obj = Path(file_path)
        intermediate = backend_instance.process(
            file_path_obj,
            format,
            extract_images=extract_images,
            images_output_dir=images_output_dir
        )

        # Convert intermediate to markdown
        markdown_converter = MarkdownConverter(intermediate)
        full_content = markdown_converter.convert()

        # Get sections and tables from index
        index_generator = IndexGenerator(intermediate)
        index_data = index_generator.generate()

        # Create a wrapper converter function for process_document
        def backend_converter(fp: str, **kwargs):
            return DocumentConverterResult(
                title=intermediate.get("metadata", {}).get("title"),
                text_content=full_content,
                metadata=intermediate.get("source", {}),
                sections=index_data.get("sections", []),
                tables=index_data.get("tables", [])
            )

        # Delegate to process_document
        result = await process_document(
            file_path=file_path,
            converter_func=backend_converter,
            converter_kwargs={},
            chunk=chunk,
            chunk_size=chunk_size,
            offset=offset,
            limit=limit,
            extract_sections=extract_sections,
            extract_tables=extract_tables,
            extract_metadata=extract_metadata,
            preview_only=preview_only,
            preview_lines=preview_lines,
            session_id=session_id,
            return_format=return_format
        )

        # Add backend info to the result
        if result.get("success"):
            result["backend_used"] = backend_instance.name
            if warnings:
                result["warnings"] = warnings

        return result

    except Exception as e:
        logger.warning(f"Backend processing failed: {e}, falling back to original logic")

    # Special case: PDF has its own implementation
    if format == "pdf":
        return await process_pdf_document(
            file_path=file_path,
            chunk=chunk,
            chunk_size=chunk_size,
            offset=offset,
            limit=limit,
            extract_sections=extract_sections,
            extract_tables=extract_tables,
            extract_metadata=extract_metadata,
            extract_images=extract_images,
            render_images=render_images,
            render_dpi=render_dpi,
            render_format=render_format,
            extract_forms=extract_forms,
            inspect_struct=inspect_struct,
            include_coords=include_coords,
            images_output_dir=images_output_dir,
            preview_only=preview_only,
            preview_lines=preview_lines,
            session_id=session_id,
            return_format=return_format,
        )

    # All other formats use process_document
    converter_kwargs = {
        "extract_metadata": extract_metadata,
        "extract_sections": extract_sections,
        "extract_tables": extract_tables,
    }

    if format == "word":
        converter_func = DocxConverter
    elif format == "excel":
        converter_func = XlsxConverter
    elif format == "ppt":
        converter_func = get_simple_converter_wrapper("ppt")
    elif format == "html":
        converter_func = HtmlConverter
    elif format == "zip":
        converter_func = get_simple_converter_wrapper("zip")
    else:
        # Unknown format - use markitdown fallback
        converter_func = get_simple_converter_wrapper("markitdown")

    return await process_document(
        file_path=file_path,
        converter_func=converter_func,
        converter_kwargs=converter_kwargs,
        chunk=chunk,
        chunk_size=chunk_size,
        offset=offset,
        limit=limit,
        extract_sections=extract_sections,
        extract_tables=extract_tables,
        extract_metadata=extract_metadata,
        preview_only=preview_only,
        preview_lines=preview_lines,
        session_id=session_id,
        return_format=return_format
    )




@mcp.tool()
async def get_supported_formats() -> dict[str, Any]:
    """List all supported file formats.

    Returns:
        Dictionary with format categories and extensions.
    """
    text_formats, binary_formats = _build_supported_format_groups()
    return {
        "text_formats": text_formats,
        "binary_formats": binary_formats,
        "tools": {
            "main": ["read_text_file", "read_binary_file"],
            "new": ["process_text_file", "process_binary_file"],
            "auxiliary": ["analyze_image", "get_vision_status", "cleanup_temp_files"]
        },
        "notes": [
            "Use read_text_file/read_binary_file for direct content access",
            "Use process_text_file/process_binary_file to save results to files",
            "File format is auto-detected by extension",
            "Explicit format parameter can override auto-detection",
        ]
    }


@mcp.tool()
async def cleanup_temp_files(
    older_than_hours: int | None = 24,
    dry_run: bool | None = False,
    cleanup_pdf_images: bool | None = True,
    cleanup_zip_extracts: bool | None = True,
    custom_directory: str | None = None
) -> dict[str, Any]:
    """Clean up temporary files created by Local Read MCP tools.

    Use this tool after completing your tasks and confirming no further need for temporary files.

    Args:
        older_than_hours: Only clean up files older than this many hours. Default: 24
        dry_run: If True, only show what would be deleted without actually deleting. Default: False
        cleanup_pdf_images: Clean up PDF image extraction directories. Default: True
        cleanup_zip_extracts: Clean up ZIP extraction directories. Default: True
        custom_directory: Optional custom directory to clean up (in addition to temp dirs)

    Returns:
        Dictionary with cleanup results including number of files/directories deleted
    """
    import shutil
    import tempfile
    from datetime import datetime

    logger = logging.getLogger(__name__)

    temp_dir = tempfile.gettempdir()
    cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600) if older_than_hours else 0

    deleted_files = 0
    deleted_dirs = 0
    errors = []
    scanned_dirs = []

    # Directories to clean up (prefix patterns)
    target_patterns = []
    if cleanup_pdf_images:
        target_patterns.append("pdf_images_")
    if cleanup_zip_extracts:
        target_patterns.append("zip_extract_")

    def is_old_enough(path: str) -> bool:
        """Check if file/directory is older than cutoff time."""
        if older_than_hours == 0:
            return True
        try:
            mtime = os.path.getmtime(path)
            return mtime < cutoff_time
        except Exception:
            return False  # If we can't get mtime, skip

    def delete_path(path: str, is_dir: bool) -> bool:
        """Delete a file or directory."""
        nonlocal deleted_files, deleted_dirs
        try:
            if dry_run:
                logger.info(f"[Dry run] Would delete: {path}")
                return True
            if is_dir:
                shutil.rmtree(path)
                deleted_dirs += 1
            else:
                os.remove(path)
                deleted_files += 1
            return True
        except Exception as e:
            error_msg = f"Failed to delete {path}: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)
            return False

    # Scan temp directory for matching directories
    try:
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path):
                # Check if directory matches any of our patterns
                for pattern in target_patterns:
                    if item.startswith(pattern):
                        scanned_dirs.append(item_path)
                        if is_old_enough(item_path):
                            delete_path(item_path, is_dir=True)
                        break
    except Exception as e:
        error_msg = f"Error scanning temp directory {temp_dir}: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    # Clean up custom directory if provided
    if custom_directory and os.path.exists(custom_directory):
        if os.path.isdir(custom_directory):
            scanned_dirs.append(custom_directory)
            if is_old_enough(custom_directory):
                delete_path(custom_directory, is_dir=True)
        else:
            if is_old_enough(custom_directory):
                delete_path(custom_directory, is_dir=False)

    # Prepare result
    result = {
        "success": True,
        "dry_run": dry_run,
        "older_than_hours": older_than_hours,
        "deleted": {
            "files": deleted_files,
            "directories": deleted_dirs,
            "total": deleted_files + deleted_dirs
        },
        "scanned_directories": scanned_dirs,
        "temp_directory": temp_dir
    }

    if errors:
        result["errors"] = errors
        result["success"] = False

    if dry_run:
        result["message"] = "Dry run completed - no files were actually deleted"
    else:
        result["message"] = f"Cleaned up {deleted_files} files and {deleted_dirs} directories"

    return result


@mcp.tool()
async def process_text_file(
    file_path: str,
    format: str | None = None,
    output_dir: str | None = None,
    backend: str = "auto"
) -> dict[str, Any]:
    """Process text-based files and save results to output directory.

    Supported formats: .txt, .md, .py, .sh, .json, .csv, .yaml, .yml

    Args:
        file_path: Path to the file to read
        format: Explicit format override (text/json/csv/yaml)
        output_dir: Directory to save output files (defaults to .local_read_mcp in current directory)
        backend: Backend to use (auto, simple)

    Returns:
        Dictionary with paths to generated files
    """
    # Detect format if not specified
    if format is None:
        format = detect_format(file_path) or "text"

    # Get backend
    registry = get_registry()

    # Parse backend type
    try:
        backend_type = BackendType(backend)
    except ValueError:
        backend_type = BackendType.AUTO

    # Select backend
    if backend_type == BackendType.AUTO:
        backend_instance = registry.select_best(format)
    else:
        backend_instance = registry.get(backend_type)

    if backend_instance is None:
        backend_instance = registry.get(BackendType.SIMPLE)

    # Validate that the selected backend supports the format
    if not backend_instance.supports_format(format):
        raise ValueError(f"Backend '{backend_instance.name}' does not support format '{format}'")

    warnings = []
    if backend_instance.warning:
        warnings.append(backend_instance.warning)

    try:
        # Create output directory
        output_manager = OutputManager(base_dir=Path(output_dir) if output_dir else None)
        output_path = output_manager.create_output_dir(file_path)

        # Process with backend
        file_path_obj = Path(file_path)
        intermediate = backend_instance.process(file_path_obj, format)

        # Save intermediate JSON
        intermediate_path = output_manager.get_output_path(output_path, "intermediate.json")
        import json
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(intermediate, f, ensure_ascii=False, indent=2)

        # Convert to markdown
        markdown_converter = MarkdownConverter(intermediate)
        markdown_content = markdown_converter.convert()
        markdown_path = output_manager.get_output_path(output_path, "output.md")
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Save index
        index_generator = IndexGenerator(intermediate)
        index_path = output_manager.get_output_path(output_path, "index.json")
        index_generator.save_to_file(str(index_path))

        return {
            "success": True,
            "output_directory": str(output_path),
            "backend_used": backend_instance.name,
            "warnings": warnings,
            "files": {
                "intermediate_json": str(intermediate_path),
                "markdown": str(markdown_path),
                "index_json": str(index_path)
            }
        }
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "backend_used": backend_instance.name,
            "warnings": warnings
        }


@mcp.tool()
async def process_binary_file(
    file_path: str,
    format: str | None = None,
    output_dir: str | None = None,
    extract_images: bool | None = None,
    backend: str = "auto"
) -> dict[str, Any]:
    """Process binary/document files and save results to output directory.

    Supported formats: .pdf, .docx, .doc, .xlsx, .xls, .pptx, .ppt, .html, .htm, .zip

    Args:
        file_path: Path to the file to read
        format: Explicit format override
        output_dir: Directory to save output files (defaults to .local_read_mcp in current directory)
        extract_images: Extract images from PDF (auto-enabled if vision is configured)
        backend: Backend to use (auto, simple)

    Returns:
        Dictionary with paths to generated files
    """
    # Detect format if not specified
    if format is None:
        format = detect_format(file_path) or "pdf"

    # Map format to backend format names
    format_mapping = {
        "pdf": "pdf",
        "word": "word",
        "excel": "excel",
        "ppt": "ppt",
        "html": "html",
        "zip": "zip"
    }
    backend_format = format_mapping.get(format, format)

    # Get backend
    registry = get_registry()

    # Parse backend type
    try:
        backend_type = BackendType(backend)
    except ValueError:
        backend_type = BackendType.AUTO

    # Select backend
    if backend_type == BackendType.AUTO:
        backend_instance = registry.select_best(backend_format)
    else:
        backend_instance = registry.get(backend_type)

    if backend_instance is None:
        backend_instance = registry.get(BackendType.SIMPLE)

    # Validate that the selected backend supports the format
    if not backend_instance.supports_format(backend_format):
        raise ValueError(f"Backend '{backend_instance.name}' does not support format '{backend_format}'")

    warnings = []
    if backend_instance.warning:
        warnings.append(backend_instance.warning)

    try:
        # Create output directory
        output_manager = OutputManager(base_dir=Path(output_dir) if output_dir else None)
        output_path = output_manager.create_output_dir(file_path)
        images_dir = output_path / "images"

        # Process with backend
        file_path_obj = Path(file_path)
        intermediate = backend_instance.process(file_path_obj, backend_format, extract_images=extract_images, images_output_dir=str(images_dir))

        # Save intermediate JSON
        intermediate_path = output_manager.get_output_path(output_path, "intermediate.json")
        import json
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(intermediate, f, ensure_ascii=False, indent=2)

        # Convert to markdown
        markdown_converter = MarkdownConverter(intermediate)
        markdown_content = markdown_converter.convert()
        markdown_path = output_manager.get_output_path(output_path, "output.md")
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Save index
        index_generator = IndexGenerator(intermediate)
        index_path = output_manager.get_output_path(output_path, "index.json")
        index_generator.save_to_file(str(index_path))

        return {
            "success": True,
            "output_directory": str(output_path),
            "backend_used": backend_instance.name,
            "warnings": warnings,
            "files": {
                "intermediate_json": str(intermediate_path),
                "markdown": str(markdown_path),
                "index_json": str(index_path),
                "images": str(images_dir) if extract_images and images_dir.exists() else None
            }
        }
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "backend_used": backend_instance.name,
            "warnings": warnings
        }


def main():
    """Main entry point for running MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Local Read MCP Server - Document processing tools")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport method: 'stdio' or 'http' (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to use when running with HTTP transport (default: 8080)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="URL path to use when running with HTTP transport (default: /mcp)",
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", port=args.port, path=args.path)


if __name__ == "__main__":
    main()
