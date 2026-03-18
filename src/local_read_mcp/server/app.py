# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
Local Read MCP Server

A Model Context Protocol server for processing various file formats.
Converts documents to markdown/text without requiring external APIs.
"""

import sys
import logging
import os
import time
import hashlib
import json
import re
import base64
from typing import Dict, Any, Optional, List
from fastmcp import FastMCP

from ..converters import (
    DocumentConverterResult,
    PdfConverter,
    DocxConverter,
    XlsxConverter,
    PptxConverter,
    HtmlConverter,
    TextConverter,
    JsonConverter,
    CsvConverter,
    YamlConverter,
    ZipConverter,
    MarkItDownConverter,
    extract_sections_from_markdown,
    apply_content_limit,
    generate_session_id,
    fix_latex_formulas,
    inspect_pdf,
    render_pdf_to_images,
    extract_form_fields,
    extract_tables,
)
from .vision import guess_mime_type_from_extension, call_vision_api
from .utils import (
    apply_pagination,
    fix_tool_arguments,
    DuplicateDetector,
    duplicate_detector,
    create_simple_converter_wrapper,
)

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("local_read_mcp-server")

# Initialize config to check if vision features should be enabled
from ..config import get_config as _get_config
_config = _get_config()
VISION_ENABLED = _config.vision_enabled

if VISION_ENABLED:
    logger.info(f"Vision features ENABLED (model: {_config.model})")
else:
    logger.info("Vision features DISABLED - configure VISION_API_KEY or OPENAI_API_KEY in .env file to enable")


def detect_format(file_path: str) -> Optional[str]:
    """Detect file format from extension.

    Returns:
        Format string or None for unknown (use markitdown fallback)
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Text formats
    if ext in ['.txt', '.md', '.py', '.sh', '.log', '.rst']:
        return 'text'
    elif ext == '.json':
        return 'json'
    elif ext == '.csv':
        return 'csv'
    elif ext in ['.yaml', '.yml']:
        return 'yaml'

    # Binary/document formats
    elif ext == '.pdf':
        return 'pdf'
    elif ext in ['.docx', '.doc']:
        return 'word'
    elif ext in ['.xlsx', '.xls']:
        return 'excel'
    elif ext in ['.pptx', '.ppt']:
        return 'ppt'
    elif ext in ['.html', '.htm']:
        return 'html'
    elif ext == '.zip':
        return 'zip'

    # Unknown - use markitdown fallback
    return None


async def process_pdf_document(
    file_path: str,
    chunk: Optional[int],
    chunk_size: Optional[int],
    offset: Optional[int],
    limit: Optional[int],
    extract_sections: Optional[bool],
    extract_tables: Optional[bool],
    extract_metadata: Optional[bool],
    extract_images: Optional[bool],
    render_images: Optional[bool],
    render_dpi: Optional[int],
    render_format: Optional[str],
    extract_forms: Optional[bool],
    inspect_struct: Optional[bool],
    include_coords: Optional[bool],
    images_output_dir: Optional[str],
    preview_only: Optional[bool],
    preview_lines: Optional[int],
    session_id: Optional[str],
    return_format: Optional[str],
) -> Dict[str, Any]:
    """Process PDF document with enhanced features.

    This is the standalone PDF processing logic extracted from the old read_pdf tool.
    """
    start_time = time.time()

    # Helper functions
    def apply_pagination(content: str, off: int, lim: Optional[int]) -> tuple[str, bool]:
        """Apply pagination to content."""
        if off >= len(content):
            return "", False
        if lim:
            end = min(off + lim, len(content))
            return content[off:end], end < len(content)
        return content[off:], False

    def fix_latex_formulas(content: str) -> str:
        """Fix common LaTeX formula parsing issues."""
        if not content:
            return content

        # Fix (cid:XXX) placeholders
        cid_map = {
            r'\(cid:16\)': '〈',
            r'\(cid:17\)': '〉',
            r'\(cid:40\)': '(',
            r'\(cid:41\)': ')',
            r'\(cid:91\)': '[',
            r'\(cid:93\)': ']',
        }
        for pattern, replacement in cid_map.items():
            content = re.sub(pattern, replacement, content)

        # Fix Greek letters
        greek_map = {
            r'\\alpha': 'α',
            r'\\beta': 'β',
            r'\\gamma': 'γ',
            r'\\delta': 'δ',
            r'\\epsilon': 'ε',
        }
        for latex_cmd, unicode_char in greek_map.items():
            content = re.sub(re.escape(latex_cmd), unicode_char, content)

        return content

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

        # Apply LaTeX fixes
        fixed_content = fix_latex_formulas(full_content)

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
        paginated_content, has_more = apply_pagination(fixed_content, char_offset, char_limit)

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
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            session_id = f"pdf_session_{file_hash}_{int(time.time())}"

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
                    "total_chunks": max(1, (len(fixed_content) + char_limit - 1) // char_limit) if char_limit else 1,
                    "chunk_size": char_limit,
                    "has_more": has_more,
                    "char_offset": char_offset,
                    "char_limit": char_limit,
                    "total_chars": len(fixed_content),
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
                total_chars = len(fixed_content)
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
                "total_chars": len(fixed_content),
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
                "content": f"Error: Failed to read PDF file: {str(e)}",
                "processing_time_ms": processing_time_ms,
            }
        else:
            return {
                "success": False,
                "error": str(e),
                "content": f"Error: Failed to read PDF file: {str(e)}",
            }


@mcp.tool()
async def analyze_image(
    image_path: str,
    question: str = "Describe this image in detail. What type of content is it?",
    api_key: Optional[str] = None
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
async def get_vision_status() -> Dict[str, Any]:
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
    converter_kwargs: Dict[str, Any],
    chunk: Optional[int] = 1,
    chunk_size: Optional[int] = 10000,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    extract_sections: Optional[bool] = False,
    extract_tables: Optional[bool] = False,
    extract_metadata: Optional[bool] = False,
    preview_only: Optional[bool] = False,
    preview_lines: Optional[int] = 100,
    session_id: Optional[str] = None,
    return_format: Optional[str] = "text"
) -> Dict[str, Any]:
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
                "content": f"Error: Failed to process file: {str(e)}",
                "processing_time_ms": processing_time_ms,
            }
        else:
            return {
                "success": False,
                "error": str(e),
                "content": f"Error: Failed to process file: {str(e)}",
            }


@mcp.tool()
async def read_text_file(
    file_path: str,
    format: Optional[str] = None,
    chunk: Optional[int] = 1,
    chunk_size: Optional[int] = 10000,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    extract_sections: Optional[bool] = False,
    extract_tables: Optional[bool] = False,
    extract_metadata: Optional[bool] = False,
    preview_only: Optional[bool] = False,
    preview_lines: Optional[int] = 100,
    session_id: Optional[str] = None,
    return_format: Optional[str] = "text"
) -> Dict[str, Any]:
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

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, pagination_info, session_id.
    """
    # Apply parameter auto-fix for common naming mistakes
    local_vars = locals().copy()
    fixed_params = fix_tool_arguments("read_text_file", local_vars)

    # Extract fixed parameters
    file_path = fixed_params.get("file_path", file_path)
    format = fixed_params.get("format", format)
    chunk = fixed_params.get("chunk", chunk)
    chunk_size = fixed_params.get("chunk_size", chunk_size)
    offset = fixed_params.get("offset", offset)
    limit = fixed_params.get("limit", limit)
    extract_sections = fixed_params.get("extract_sections", extract_sections)
    extract_tables = fixed_params.get("extract_tables", extract_tables)
    extract_metadata = fixed_params.get("extract_metadata", extract_metadata)
    preview_only = fixed_params.get("preview_only", preview_only)
    preview_lines = fixed_params.get("preview_lines", preview_lines)
    session_id = fixed_params.get("session_id", session_id)
    return_format = fixed_params.get("return_format", return_format)

    # Auto-detect format if not provided
    if not format:
        format = detect_format(file_path)

    # Map format to converter
    converter_func = None
    converter_kwargs = {
        "extract_metadata": extract_metadata,
        "extract_sections": extract_sections,
        "extract_tables": extract_tables,
    }

    if format == "text":
        converter_func = create_simple_converter_wrapper(TextConverter, "text")
    elif format == "json":
        converter_func = create_simple_converter_wrapper(JsonConverter, "json")
    elif format == "csv":
        converter_func = create_simple_converter_wrapper(CsvConverter, "csv")
    elif format == "yaml":
        converter_func = create_simple_converter_wrapper(YamlConverter, "yaml")
    else:
        # Unknown format - use markitdown fallback
        converter_func = create_simple_converter_wrapper(MarkItDownConverter, "markitdown")

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
    format: Optional[str] = None,
    # Standard pagination
    chunk: Optional[int] = 1,
    chunk_size: Optional[int] = 10000,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    # Standard structured extraction
    extract_sections: Optional[bool] = False,
    extract_tables: Optional[bool] = False,
    extract_metadata: Optional[bool] = False,
    preview_only: Optional[bool] = False,
    preview_lines: Optional[int] = 100,
    session_id: Optional[str] = None,
    return_format: Optional[str] = "text",
    # PDF-specific features (only used when format=pdf or auto-detected as pdf)
    extract_images: Optional[bool] = False,
    render_images: Optional[bool] = False,
    render_dpi: Optional[int] = 200,
    render_format: Optional[str] = "png",
    extract_forms: Optional[bool] = False,
    inspect_struct: Optional[bool] = False,
    include_coords: Optional[bool] = False,
    images_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
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
        extract_images: Extract images from PDF. Saves images to output directory and returns image info. Default: False.
        render_images: Render PDF pages to images. Default: False.
        render_dpi: DPI for rendered images. Default: 200.
        render_format: Format for rendered images (png or jpeg). Default: png.
        extract_forms: Extract form fields from PDF. Default: False.
        inspect_struct: Get complete PDF structure (metadata, outline, fonts, etc.). Default: False.
        include_coords: Include text with bounding box coordinates. Default: False.
        images_output_dir: Directory to save extracted/rendered images. If None, uses temporary directory. Default: None.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, pagination_info, pdf_pages, images, session_id.
    """
    # Auto-enable extract_images if vision is configured and not explicitly set
    if extract_images is None and VISION_ENABLED:
        extract_images = True
        logger.info("Vision enabled: auto-enabling extract_images=True for PDF")

    # Apply parameter auto-fix for common naming mistakes
    local_vars = locals().copy()
    fixed_params = fix_tool_arguments("read_binary_file", local_vars)

    # Extract fixed parameters
    file_path = fixed_params.get("file_path", file_path)
    format = fixed_params.get("format", format)
    chunk = fixed_params.get("chunk", chunk)
    chunk_size = fixed_params.get("chunk_size", chunk_size)
    offset = fixed_params.get("offset", offset)
    limit = fixed_params.get("limit", limit)
    extract_sections = fixed_params.get("extract_sections", extract_sections)
    extract_tables = fixed_params.get("extract_tables", extract_tables)
    extract_metadata = fixed_params.get("extract_metadata", extract_metadata)
    extract_images = fixed_params.get("extract_images", extract_images)
    render_images = fixed_params.get("render_images", render_images)
    render_dpi = fixed_params.get("render_dpi", render_dpi)
    render_format = fixed_params.get("render_format", render_format)
    extract_forms = fixed_params.get("extract_forms", extract_forms)
    inspect_struct = fixed_params.get("inspect_struct", inspect_struct)
    include_coords = fixed_params.get("include_coords", include_coords)
    images_output_dir = fixed_params.get("images_output_dir", images_output_dir)
    preview_only = fixed_params.get("preview_only", preview_only)
    preview_lines = fixed_params.get("preview_lines", preview_lines)
    session_id = fixed_params.get("session_id", session_id)
    return_format = fixed_params.get("return_format", return_format)

    # Auto-detect format if not provided
    if not format:
        format = detect_format(file_path)

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
    converter_func = None
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
        converter_func = create_simple_converter_wrapper(PptxConverter, "pptx")
    elif format == "html":
        converter_func = HtmlConverter
    elif format == "zip":
        converter_func = create_simple_converter_wrapper(ZipConverter, "zip")
    else:
        # Unknown format - use markitdown fallback
        converter_func = create_simple_converter_wrapper(MarkItDownConverter, "markitdown")

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


# ============================================
# Deprecated tools - for backward compatibility
# ============================================

@mcp.tool()
async def read_pdf(file_path: str, **kwargs):
    """Deprecated: Use read_binary_file instead.

    This is a backward compatibility alias for read_binary_file(format="pdf").
    """
    logger.warning("read_pdf is deprecated, use read_binary_file instead (or read_binary_file(format='pdf'))")
    return await read_binary_file(file_path, format="pdf", **kwargs)


@mcp.tool()
async def read_word(file_path: str, **kwargs):
    """Deprecated: Use read_binary_file instead.

    This is a backward compatibility alias for read_binary_file(format="word").
    """
    logger.warning("read_word is deprecated, use read_binary_file instead (or read_binary_file(format='word'))")
    return await read_binary_file(file_path, format="word", **kwargs)


@mcp.tool()
async def read_excel(file_path: str, **kwargs):
    """Deprecated: Use read_binary_file instead.

    This is a backward compatibility alias for read_binary_file(format="excel").
    """
    logger.warning("read_excel is deprecated, use read_binary_file instead (or read_binary_file(format='excel'))")
    return await read_binary_file(file_path, format="excel", **kwargs)


@mcp.tool()
async def read_powerpoint(file_path: str, **kwargs):
    """Deprecated: Use read_binary_file instead.

    This is a backward compatibility alias for read_binary_file(format="ppt").
    """
    logger.warning("read_powerpoint is deprecated, use read_binary_file instead (or read_binary_file(format='ppt'))")
    return await read_binary_file(file_path, format="ppt", **kwargs)


@mcp.tool()
async def read_html(file_path: str, **kwargs):
    """Deprecated: Use read_binary_file instead.

    This is a backward compatibility alias for read_binary_file(format="html").
    """
    logger.warning("read_html is deprecated, use read_binary_file instead (or read_binary_file(format='html'))")
    return await read_binary_file(file_path, format="html", **kwargs)


@mcp.tool()
async def read_text(file_path: str, **kwargs):
    """Deprecated: Use read_text_file instead.

    This is a backward compatibility alias for read_text_file(format="text").
    """
    logger.warning("read_text is deprecated, use read_text_file instead (or read_text_file(format='text'))")
    return await read_text_file(file_path, format="text", **kwargs)


@mcp.tool()
async def read_json(file_path: str, **kwargs):
    """Deprecated: Use read_text_file instead.

    This is a backward compatibility alias for read_text_file(format="json").
    """
    logger.warning("read_json is deprecated, use read_text_file instead (or read_text_file(format='json'))")
    return await read_text_file(file_path, format="json", **kwargs)


@mcp.tool()
async def read_csv(file_path: str, **kwargs):
    """Deprecated: Use read_text_file instead.

    This is a backward compatibility alias for read_text_file(format="csv").
    """
    logger.warning("read_csv is deprecated, use read_text_file instead (or read_text_file(format='csv'))")
    return await read_text_file(file_path, format="csv", **kwargs)


@mcp.tool()
async def read_yaml(file_path: str, **kwargs):
    """Deprecated: Use read_text_file instead.

    This is a backward compatibility alias for read_text_file(format="yaml").
    """
    logger.warning("read_yaml is deprecated, use read_text_file instead (or read_text_file(format='yaml'))")
    return await read_text_file(file_path, format="yaml", **kwargs)


@mcp.tool()
async def read_zip(file_path: str, **kwargs):
    """Deprecated: Use read_binary_file instead.

    This is a backward compatibility alias for read_binary_file(format="zip").
    """
    logger.warning("read_zip is deprecated, use read_binary_file instead (or read_binary_file(format='zip'))")
    return await read_binary_file(file_path, format="zip", **kwargs)


@mcp.tool()
async def read_with_markitdown(file_path: str, **kwargs):
    """Deprecated: Use read_text_file or read_binary_file instead.

    This is a backward compatibility alias that uses markitdown fallback.
    """
    logger.warning("read_with_markitdown is deprecated, use read_text_file or read_binary_file instead")
    # Try to detect and use appropriate tool
    ext = os.path.splitext(file_path)[1].lower()
    text_exts = ['.txt', '.md', '.py', '.sh', '.log', '.rst', '.json', '.csv', '.yaml', '.yml']
    if ext in text_exts:
        return await read_text_file(file_path, **kwargs)
    else:
        return await read_binary_file(file_path, **kwargs)


@mcp.tool()
@mcp.tool()
async def get_supported_formats() -> Dict[str, Any]:
    """Get a list of all supported file formats.

    Returns:
        A dictionary listing all supported file extensions.
    """
    return {
        "success": True,
        "documents": {
            "pdf": "PDF documents (.pdf)",
            "docx": "Word documents (.docx)",
            "xlsx": "Excel spreadsheets (xlsx) - converted to markdown tables",
            "pptx": "PowerPoint presentations (pptx)",
            "html": "HTML files (html, .htm)",
        },
        "text": {
            "txt": "Plain text files (txt)",
            "md": "Markdown files (md)",
            "json": "JSON files (json)",
            "yaml": "YAML files (yaml, .yml)",
            "csv": "CSV files (csv) - converted to markdown tables",
            "toml": "TOML files (toml)",
            "py": "Python files (py)",
            "sh": "Shell scripts (sh)",
        },
        "archives": {
            "zip": "ZIP archives (zip) - lists contents and extracts files",
        },
        "fallback": {
            "markitdown": "MarkItDown fallback - supports many additional formats",
        },
    }


@mcp.tool()
async def cleanup_temp_files(
    older_than_hours: Optional[int] = 24,
    dry_run: Optional[bool] = False,
    cleanup_pdf_images: Optional[bool] = True,
    cleanup_zip_extracts: Optional[bool] = True,
    custom_directory: Optional[str] = None
) -> Dict[str, Any]:
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
