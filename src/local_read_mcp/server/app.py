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

def get_simple_converter_wrapper(format_name: str) -> Callable[..., Any]:
    """Get (and cache) wrapper for simple converters."""
    if format_name in _SIMPLE_CONVERTER_CACHE:
        return _SIMPLE_CONVERTER_CACHE[format_name]

    converter_func, converter_name = _SIMPLE_CONVERTER_BUILDERS[format_name]
    wrapper = create_simple_converter_wrapper(converter_func, converter_name)
    _SIMPLE_CONVERTER_CACHE[format_name] = wrapper
    return wrapper


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
) -> dict[str, Any]:
    """Analyze an image using OpenAI-compatible vision API and save result to .local_read_mcp/analysis/.

    Args:
        image_path: Path to the image file to analyze
        question: Question to ask about the image
        api_key: API key (overrides config if provided)

    Returns:
        Dict with analysis result and saved file path

    Environment Variables (.env):
        VISION_API_KEY: Your API key (or OPENAI_API_KEY)
        VISION_BASE_URL: API base URL (or OPENAI_BASE_URL)
        VISION_MODEL: Model name (or OPENAI_VISION_MODEL, default: gpt-4o)
        VISION_MAX_IMAGE_SIZE_MB: Max image size in MB (default: 20)
    """
    if not VISION_ENABLED:
        return {"success": False, "error": "Vision is not enabled. Set VISION_API_KEY or OPENAI_API_KEY in .env file."}

    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}"}

    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    max_size = _config.vision_max_image_size_mb
    if file_size_mb > max_size:
        return {"success": False, "error": f"Image too large ({file_size_mb:.2f}MB). Maximum: {max_size}MB"}

    effective_api_key = api_key or _config.api_key
    if not effective_api_key:
        return {"success": False, "error": "API key not configured. Set VISION_API_KEY or OPENAI_API_KEY in .env."}

    # Call vision API
    result_text = await call_vision_api(
        image_path=image_path,
        question=question,
        api_key=effective_api_key,
        base_url=_config.base_url,
        model=_config.model
    )

    # Save to .local_read_mcp/analysis/ in the working directory
    from pathlib import Path as _Path
    analysis_dir = _Path.cwd() / ".local_read_mcp" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    image_name = _Path(image_path).stem
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in image_name)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    result_filename = f"{safe_name}_{timestamp}.md"
    result_path = analysis_dir / result_filename

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"# Image Analysis: {image_name}\n\n")
        f.write(f"- **Source**: `{image_path}`\n")
        f.write(f"- **Question**: {question}\n")
        f.write(f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Analysis\n\n")
        f.write(result_text)
        f.write("\n")

    return {
        "success": True,
        "analysis": result_text,
        "saved_path": str(result_path),
    }


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
async def process_binary_file(
    file_path: str,
    format: str | None = None,
    backend: str = "auto",
    # Page range control
    chapter_split: bool | str | int = "auto",
    start_page: int | None = None,
    end_page: int | None = None,
    page_batch_size: int = 64,
    # PDF-specific
    extract_images: bool | None = None,
    render_images: bool = False,
    render_dpi: int = 200,
    render_format: str = "png",
    extract_forms: bool = False,
    inspect_struct: bool = False,
    include_coords: bool = False,
) -> dict[str, Any]:
    """MUST use this tool for ANY binary/document file before reading it.

    This includes PDF, Word, Excel, PowerPoint, HTML, ZIP, images, and any
    file that is not plain text. The built-in Read tool cannot handle these
    formats properly; processing them through this tool first is required.

    Results are saved in .local_read_mcp/<file>_<timestamp>/
    with intermediate.json, output.md and index.json.
    For large PDFs, chapter_split detects sections and processes each chunk
    independently, then merges the results.

    Args:
        file_path: Path to the file to process.
        format: Override auto-detected format.
        backend: Backend (auto/simple/vlm-hybrid). Default: auto.
        chapter_split: "auto" (split >30p), "chapter", N (fixed pages), False (off).
        start_page: 0-based start page.
        end_page: 0-based end page.
        page_batch_size: Pages per batch (default: 64).
        extract_images: Extract images from PDF (auto if vision configured).
        render_images: Render PDF pages to images.
        render_dpi: Render DPI (default: 200).
        render_format: png or jpeg (default: png).
        extract_forms: Extract PDF form fields.
        inspect_struct: Get PDF structure metadata.
        include_coords: Include text with bounding boxes.
    """
    # ── 1. Format detection ──────────────────────────────────────
    if format is None:
        format = detect_format(file_path)
    if format is None:
        format = "text"

    # Auto-enable extract_images for PDF if vision is configured
    if format == "pdf" and extract_images is None:
        extract_images = bool(VISION_ENABLED)

    # ── 2. Backend selection ─────────────────────────────────────
    registry = get_registry()
    try:
        backend_type = BackendType(backend)
    except ValueError:
        backend_type = BackendType.AUTO

    if backend_type == BackendType.AUTO:
        backend_instance = registry.select_best(format)
    else:
        backend_instance = registry.get(backend_type)

    if backend_instance is None:
        backend_instance = registry.get(BackendType.SIMPLE)

    if not backend_instance.supports_format(format):
        raise ValueError(
            f"Backend '{backend_instance.name}' does not support format '{format}'"
        )

    warnings = []
    if backend_instance.warning:
        warnings.append(backend_instance.warning)

    # ── 3. Plan chunks (segmenter integration) ───────────────────
    chunks = _plan_chunks(
        file_path=file_path,
        format=format,
        backend_name=backend_instance.name,
        chapter_split=chapter_split,
        start_page=start_page,
        end_page=end_page,
    )

    # ── 4. Create output directory ───────────────────────────────
    output_manager = OutputManager()
    output_path = output_manager.create_output_dir(file_path)
    images_dir = output_path / "images"

    # ── 5. Build backend kwargs ──────────────────────────────────
    backend_kwargs: dict[str, Any] = {}
    if format == "pdf":
        backend_kwargs["extract_images"] = extract_images
        backend_kwargs["images_output_dir"] = str(images_dir)
        backend_kwargs["render_images"] = render_images
        backend_kwargs["render_dpi"] = render_dpi
        backend_kwargs["render_format"] = render_format
        backend_kwargs["extract_forms"] = extract_forms
        backend_kwargs["inspect_struct"] = inspect_struct
        backend_kwargs["include_coords"] = include_coords

    # ── 6. Process ───────────────────────────────────────────────
    try:
        if len(chunks) == 1:
            # Single-chunk: process in-place (same as before)
            result = _process_and_save(
                file_path=file_path,
                backend=backend_instance,
                format=format,
                output_path=output_path,
                images_dir=images_dir,
                chunk=chunks[0],
                backend_kwargs=backend_kwargs,
            )
            result["success"] = True
            result["backend_used"] = backend_instance.name
            result["output_directory"] = str(output_path)
            result["files"] = {
                "intermediate_json": str(result["intermediate_path"]),
                "markdown": str(result["markdown_path"]),
                "index_json": str(result["index_path"]),
            }
            if "image_files" in result:
                result["files"]["images"] = str(result["image_files"][0].parent)
                result["image_count"] = len(result["image_files"])
            if warnings:
                result["warnings"] = warnings
            return result

        # Multi-chunk: process each chunk independently, then merge
        chunk_results: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            chunk_dir = output_path / f"chunk_{idx + 1:04d}"
            chunk_dir.mkdir(exist_ok=True)
            try:
                cr = _process_and_save(
                    file_path=file_path,
                    backend=backend_instance,
                    format=format,
                    output_path=chunk_dir,
                    images_dir=chunk_dir / "images",
                    chunk=chunk,
                    backend_kwargs=backend_kwargs,
                )
                chunk_results.append(cr)
            except Exception as e:
                logger.error("Chunk %d (%s) failed: %s", idx + 1, chunk.title, e)
                chunk_results.append({
                    "error": str(e),
                    "title": chunk.title,
                    "phys_start": chunk.phys_start,
                    "phys_end": chunk.phys_end,
                })

        # Merge and save structural TOC
        _save_structural_toc(output_path, chunks)

        # Merge chunk markdowns into a single output.md
        merged_md = _merge_chunk_markdowns(chunk_results)
        if merged_md:
            merged_md_path = output_path / "output.md"
            with open(merged_md_path, 'w', encoding='utf-8') as f:
                f.write(merged_md)

        succeeded = [cr for cr in chunk_results if "error" not in cr]
        chunk_files = []
        for cr in chunk_results:
            info = {"title": cr.get("title", ""), "phys_start": cr.get("phys_start"), "phys_end": cr.get("phys_end")}
            if "error" in cr:
                info["error"] = cr["error"]
            else:
                info["intermediate_json"] = str(cr["intermediate_path"])
                info["markdown"] = str(cr["markdown_path"])
            chunk_files.append(info)

        return {
            "success": True,
            "output_directory": str(output_path),
            "backend_used": backend_instance.name,
            "chunk_count": len(chunks),
            "files": {
                "chunks": chunk_files,
                "structural_toc": str(output_path / "structural_toc.json") if (output_path / "structural_toc.json").exists() else None,
                "merged_markdown": str(merged_md_path) if merged_md else None,
            },
            "warnings": warnings,
        }

    except Exception as e:
        logger.error("Processing failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "backend_used": backend_instance.name,
        }


# ── Chunk planning ─────────────────────────────────────────────────


def _plan_chunks(
    file_path: str,
    format: str,
    backend_name: str,
    chapter_split: bool | str | int,
    start_page: int | None,
    end_page: int | None,
) -> list[Any]:
    """Determine processing chunks for the given document.

    Returns a list of Chunk objects (from the segmenter module).
    A single-element list means no splitting.
    """
    from ..segmenter import Chunk, ChunkPlanner, TocExtractor  # noqa: PLC0415

    # No splitting requested
    if chapter_split is False or chapter_split is None:
        return [Chunk(phys_start=start_page or 0, phys_end=end_page or 2**31 - 1)]

    # Only PDF + layout-capable backend triggers the segmenter
    if format != "pdf":
        return [Chunk(phys_start=start_page or 0, phys_end=end_page or 2**31 - 1)]

    # Load document for page count and chapter detection
    try:
        import fitz  # noqa: PLC0415
    except ImportError:
        logger.warning("PyMuPDF not available, cannot detect chapters")
        return [Chunk(phys_start=start_page or 0, phys_end=end_page or 2**31 - 1)]

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.warning("Cannot open PDF for chapter detection: %s, processing whole file", e)
        s = start_page or 0
        e = end_page or 2**31 - 1
        return [Chunk(phys_start=s, phys_end=e)]

    total = doc.page_count

    # Determine if splitting is worthwhile
    need_split = False
    split_type: str | int = "auto"
    if isinstance(chapter_split, str) and chapter_split == "auto":
        need_split = total > 30
        split_type = "auto"
    elif isinstance(chapter_split, str) and chapter_split in ("chapter", "section"):
        need_split = True
        split_type = chapter_split
    elif isinstance(chapter_split, int):
        need_split = True
        split_type = chapter_split

    if not need_split:
        doc.close()
        s = start_page or 0
        e = min(end_page or total - 1, total - 1)
        return [Chunk(phys_start=s, phys_end=e)]

    # Run segmenter
    try:
        extractor = TocExtractor()
        chapters = extractor.extract(doc)
        planner = ChunkPlanner(overlap=2)

        if chapters:
            raw_chunks = planner.plan_from_chapters(chapters, total_pages=total)
        elif isinstance(split_type, int):
            raw_chunks = planner.plan_fixed(total, chunk_size=split_type)
        else:
            raw_chunks = planner.plan_fixed(total, chunk_size=20)

        doc.close()
    except Exception as e:
        logger.warning("Chapter detection failed: %s, falling back to fixed chunks", e)
        doc.close()
        planner = ChunkPlanner()
        raw_chunks = planner.plan_fixed(total, chunk_size=20)

    # Apply start_page / end_page bounds
    if start_page is not None or end_page is not None:
        bounded: list[Any] = []
        for c in raw_chunks:
            s = c.phys_start
            e = c.phys_end
            if start_page is not None:
                s = max(s, start_page)
            if end_page is not None:
                e = min(e, end_page)
            if s <= e:
                bounded.append(Chunk(phys_start=s, phys_end=e, title=c.title, level=c.level, batch_size=c.batch_size))
        return bounded

    return raw_chunks


# ── Single-chunk processing + save ─────────────────────────────────


def _process_and_save(
    file_path: str,
    backend,
    format: str,
    output_path: Path,
    images_dir: Path,
    chunk: Any,
    backend_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Process one chunk through the backend and save all outputs."""
    import json  # noqa: PLC0415

    # For PDF page-range chunks, slice the PDF into a temp file
    # so the backend receives a PDF that starts at page 0.
    if format == "pdf":
        import fitz  # noqa: PLC0415

        try:
            src = fitz.open(file_path)
        except Exception:
            # Not a real PDF — pass the file as-is, let the backend handle it
            sliced_path = Path(file_path)
        else:
            import tempfile  # noqa: PLC0415
            total = src.page_count
            p_start = max(0, min(chunk.phys_start, total - 1))
            p_end = max(p_start, min(chunk.phys_end, total - 1))

            if p_start == 0 and p_end >= total - 1:
                sliced_path = Path(file_path)
                src.close()
            else:
                sliced = fitz.open()
                sliced.insert_pdf(src, from_page=p_start, to_page=p_end)
                if images_dir:
                    images_dir.mkdir(parents=True, exist_ok=True)
                tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".pdf", dir=str(images_dir.parent) if images_dir else None)
                os.close(tmp_fd)
                sliced.save(tmp_path_str)
                sliced.close()
                sliced_path = Path(tmp_path_str)
                src.close()
    else:
        sliced_path = Path(file_path)

    try:
        intermediate = backend.process(sliced_path, format, **backend_kwargs)
    finally:
        # Clean up temp slice if created
        if format == "pdf" and sliced_path != Path(file_path) and sliced_path.exists():
            sliced_path.unlink(missing_ok=True)

    # Save intermediate.json
    intermediate_path = output_path / "intermediate.json"
    with open(intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(intermediate, f, ensure_ascii=False, indent=2)

    # Save output.md
    markdown_converter = MarkdownConverter(intermediate)
    markdown_content = markdown_converter.convert()
    markdown_path = output_path / "output.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    # Save index.json
    index_generator = IndexGenerator(intermediate)
    index_path = output_path / "index.json"
    index_generator.save_to_file(str(index_path))

    result: dict[str, Any] = {
        "title": chunk.title if hasattr(chunk, 'title') else "",
        "phys_start": chunk.phys_start if hasattr(chunk, 'phys_start') else 0,
        "phys_end": chunk.phys_end if hasattr(chunk, 'phys_end') else 0,
        "intermediate_path": intermediate_path,
        "markdown_path": markdown_path,
        "index_path": index_path,
        "intermediate": intermediate,
        "markdown_content": markdown_content,
    }

    if images_dir.exists():
        image_files = list(images_dir.iterdir())
        if image_files:
            result["image_files"] = image_files

    return result


# ── Merging helpers ────────────────────────────────────────────────


def _merge_chunk_markdowns(chunk_results: list[dict[str, Any]]) -> str:
    """Concatenate chunk markdowns with chapter separators."""
    parts: list[str] = []
    for cr in chunk_results:
        if "error" in cr:
            parts.append(
                f"\n\n---\n## [{cr.get('title', 'error')}] (processing failed)\n\n"
                f"Error: {cr['error']}\n"
            )
            continue
        md = cr.get("markdown_content", "")
        title = cr.get("title", "")
        p_start = cr.get("phys_start", 0)
        p_end = cr.get("phys_end", 0)
        header = f"\n\n---\n# {title}  (pages {p_start + 1}–{p_end + 1})\n\n"
        parts.append(header + md)
    return "\n".join(parts).strip()


def _save_structural_toc(output_path: Path, chunks: list[Any]) -> None:
    """Save the structural table of contents as JSON."""
    import json  # noqa: PLC0415

    toc_data = []
    for idx, c in enumerate(chunks):
        toc_data.append({
            "chunk_index": idx + 1,
            "title": getattr(c, "title", ""),
            "level": getattr(c, "level", 1),
            "phys_start": getattr(c, "phys_start", 0),
            "phys_end": getattr(c, "phys_end", 0),
        })

    toc_path = output_path / "structural_toc.json"
    with open(toc_path, 'w', encoding='utf-8') as f:
        json.dump(toc_data, f, ensure_ascii=False, indent=2)


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
