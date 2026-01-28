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

from .converters import (
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
)

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("local_read_mcp-server")

# Initialize config to check if vision features should be enabled
from .config import get_config as _get_config
_config = _get_config()
VISION_ENABLED = _config.vision_enabled

if VISION_ENABLED:
    logger.info(f"Vision features ENABLED (model: {_config.model})")
else:
    logger.info("Vision features DISABLED - configure VISION_API_KEY or OPENAI_API_KEY in .env file to enable")


# Vision helper functions
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


# Helper functions for unified document processing
def apply_pagination(content: str, offset: int, limit: Optional[int]) -> tuple[str, bool]:
    """Apply pagination to content."""
    if offset >= len(content):
        return "", False
    if limit:
        end = min(offset + limit, len(content))
        return content[offset:end], end < len(content)
    return content[offset:], False


def generate_session_id(file_path: str, prefix: str = "session") -> str:
    """Generate a unique session ID for a file."""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    timestamp = int(time.time())
    return f"{prefix}_{file_hash}_{timestamp}"


def fix_tool_arguments(tool_name: str, arguments: dict) -> dict:
    """
    Auto-fix common parameter name mistakes made by LLM.

    This function automatically corrects common parameter naming errors that LLMs
    make when calling tools, improving the success rate of tool calls.

    Args:
        tool_name: Name of the tool being called
        arguments: Original arguments dictionary

    Returns:
        Fixed arguments dictionary with corrected parameter names

    Examples:
        >>> fix_tool_arguments("read_pdf", {"page": 1, "page_size": 5000})
        {"chunk": 1, "chunk_size": 5000}

        >>> fix_tool_arguments("read_pdf", {"filepath": "/path/to/file.pdf"})
        {"file_path": "/path/to/file.pdf"}
    """
    fixed = arguments.copy()

    # Historical parameter name compatibility (page -> chunk migration)
    renames = {
        "page": "chunk",
        "page_size": "chunk_size",
        "pagesize": "chunk_size",
        "pages": "chunk",
        # Common file path variations
        "filepath": "file_path",
        "path": "file_path",
        "file": "file_path",
        "filename": "file_path",
        # Format variations
        "format": "return_format",
        "output_format": "return_format",
        # Other common mistakes
        "preview": "preview_only",
        "metadata": "extract_metadata",
        "sections": "extract_sections",
        "tables": "extract_tables",
        "images": "extract_images",
        "image_dir": "images_output_dir",
        "output_dir": "images_output_dir",
    }

    for old_name, new_name in renames.items():
        if old_name in fixed and new_name not in fixed:
            fixed[new_name] = fixed.pop(old_name)
            logger.info(f"[Parameter Auto-Fix] {tool_name}: '{old_name}' → '{new_name}'")

    return fixed


class DuplicateDetector:
    """
    Detect duplicate document read requests to prevent infinite loops.

    This class tracks document read requests per session and warns when the same
    file+chunk combination is requested multiple times, which may indicate that
    an agent is stuck in a loop.

    Attributes:
        max_repeats: Maximum number of times a chunk can be requested before warning
        request_cache: Dictionary mapping session_id to file+chunk request counts
    """

    def __init__(self, max_repeats: int = 3):
        """
        Initialize the duplicate detector.

        Args:
            max_repeats: Maximum allowed repeats before warning (default: 3)
        """
        self.max_repeats = max_repeats
        self.request_cache: Dict[str, Dict[str, int]] = {}

    def check_and_record(
        self,
        session_id: str,
        file_path: str,
        chunk: int,
        chunk_size: int
    ) -> Optional[str]:
        """
        Check if this is a duplicate request and record it.

        Args:
            session_id: Session identifier
            file_path: Path to the file being read
            chunk: Chunk number being requested
            chunk_size: Size of the chunk

        Returns:
            Warning message if duplicate detected, None otherwise

        Examples:
            >>> detector = DuplicateDetector(max_repeats=3)
            >>> detector.check_and_record("sess1", "/file.pdf", 1, 10000)
            None  # First request, no warning

            >>> # After 3 more requests for the same chunk...
            >>> detector.check_and_record("sess1", "/file.pdf", 1, 10000)
            "Warning: This chunk has been requested 3 times..."
        """
        # Create cache key: file_path + chunk + chunk_size
        cache_key = f"{file_path}:chunk{chunk}:size{chunk_size}"

        # Initialize session cache if needed
        if session_id not in self.request_cache:
            self.request_cache[session_id] = {}

        # Get current count
        count = self.request_cache[session_id].get(cache_key, 0)

        # Check if exceeded max repeats
        if count >= self.max_repeats:
            warning_msg = (
                f"Warning: Chunk {chunk} of '{os.path.basename(file_path)}' has been "
                f"requested {count} times in this session. You may be in a loop. "
                f"Suggestions: (1) Check if has_more=False, (2) Try different chunk numbers, "
                f"(3) Use preview_only=True to assess content first."
            )
            logger.warning(f"[Duplicate Detection] {warning_msg}")
            # Still record the request
            self.request_cache[session_id][cache_key] = count + 1
            return warning_msg

        # Record the request
        self.request_cache[session_id][cache_key] = count + 1

        # Log for debugging (only for repeat requests)
        if count > 0:
            logger.info(
                f"[Duplicate Detection] Chunk {chunk} of '{os.path.basename(file_path)}' "
                f"requested {count + 1} times in session {session_id[:8]}..."
            )

        return None

    def clear_session(self, session_id: str):
        """
        Clear cache for a specific session.

        Args:
            session_id: Session identifier to clear
        """
        if session_id in self.request_cache:
            del self.request_cache[session_id]
            logger.info(f"[Duplicate Detection] Cleared session {session_id[:8]}...")

    def get_session_stats(self, session_id: str) -> Dict[str, int]:
        """
        Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary mapping cache keys to request counts
        """
        return self.request_cache.get(session_id, {}).copy()


# Global duplicate detector instance
duplicate_detector = DuplicateDetector(max_repeats=3)


def create_simple_converter_wrapper(converter_func, converter_name: str = ""):
    """Create a wrapper for simple converters that don't support enhanced parameters.

    Args:
        converter_func: The original converter function
        converter_name: Name of the converter for logging

    Returns:
        A wrapper function that supports extract_metadata, extract_sections, extract_tables
    """
    def wrapper(file_path: str, **kwargs) -> DocumentConverterResult:
        # Call the original converter function
        result = converter_func(file_path)

        # Apply content limit (200,000 characters) like other converters
        result.text_content = apply_content_limit(result.text_content)

        # If extract_metadata is requested, add file metadata
        if kwargs.get('extract_metadata', False):
            result.metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                "file_extension": os.path.splitext(file_path)[1],
            }

        # If extract_sections is requested, extract sections from markdown
        if kwargs.get('extract_sections', False):
            result.sections = extract_sections_from_markdown(result.text_content)

        # If extract_tables is requested
        if kwargs.get('extract_tables', False):
            result.tables = []  # No table extraction for simple converters

        return result

    return wrapper


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
async def read_pdf(
    file_path: str,
    chunk: Optional[int] = 1,
    chunk_size: Optional[int] = 10000,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    extract_sections: Optional[bool] = False,
    extract_tables: Optional[bool] = False,
    extract_metadata: Optional[bool] = False,
    extract_images: Optional[bool] = False,
    images_output_dir: Optional[str] = None,
    preview_only: Optional[bool] = False,
    preview_lines: Optional[int] = 100,
    session_id: Optional[str] = None,
    return_format: Optional[str] = "text"
) -> Dict[str, Any]:
    """Read and convert a PDF file to markdown text with LaTeX formula fixing.

    USAGE STRATEGY:
    - For unknown file size: Start with preview_only=True, preview_lines=100 to assess content
    - For normal reading: Use chunk=1, chunk_size=10000 (default)
    - For structured extraction (extract_sections=True, return_format="json"): Use smaller chunk_size=5000-8000 to avoid token limits
    - For academic papers: LaTeX formulas are automatically fixed (CID placeholders, Greek letters, math symbols)
    - For multi-chunk reading: Reuse session_id from previous response for better performance
    - For PDF with images: Set extract_images=True to extract all images from PDF (requires PyMuPDF)
    - IMPORTANT: Both text and json formats include has_more flag and pagination info
    - When has_more=True, continue reading with chunk=chunk+1 until has_more=False
    - Note: "chunk" parameter divides content by character count, "pdf_pages" shows actual PDF page count
    - WARNING: Large chunk_size (>10k) with extract_sections=True may exceed token limits in JSON format

    Args:
        file_path: The path to the PDF file to read
        chunk: Chunk number for content pagination (1-indexed). Default: 1.
        chunk_size: Number of characters per chunk. Default: 10000.
        offset: Character offset (alternative to chunk). If specified, overrides chunk.
        limit: Character limit (alternative to chunk_size). If specified, overrides chunk_size.
        extract_sections: Extract document sections/headings. Use for structured documents. Default: False.
        extract_tables: Extract table information. Default: False.
        extract_metadata: Extract file metadata (size, path, timestamp, PDF pages). Use with return_format="json". Default: False.
        extract_images: Extract images from PDF. Saves images to output directory and returns image info. Default: False.
        images_output_dir: Directory to save extracted images. If None, uses temporary directory. Default: None.
        preview_only: Return only first N lines without full conversion. Use for quick assessment. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Reuse for consecutive chunk requests.
        return_format: Output format: 'json' (structured with metadata/sections) or 'text' (plain). Default: 'text'.

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
    fixed_params = fix_tool_arguments("read_pdf", local_vars)

    # Extract fixed parameters
    file_path = fixed_params.get("file_path", file_path)
    chunk = fixed_params.get("chunk", chunk)
    chunk_size = fixed_params.get("chunk_size", chunk_size)
    offset = fixed_params.get("offset", offset)
    limit = fixed_params.get("limit", limit)
    extract_sections = fixed_params.get("extract_sections", extract_sections)
    extract_tables = fixed_params.get("extract_tables", extract_tables)
    extract_metadata = fixed_params.get("extract_metadata", extract_metadata)
    extract_images = fixed_params.get("extract_images", extract_images)
    images_output_dir = fixed_params.get("images_output_dir", images_output_dir)
    preview_only = fixed_params.get("preview_only", preview_only)
    preview_lines = fixed_params.get("preview_lines", preview_lines)
    session_id = fixed_params.get("session_id", session_id)
    return_format = fixed_params.get("return_format", return_format)

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
            r'\(cid:16\)': '〈',
            r'\(cid:17\)': '〉',
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

    def extract_sections_from_text(content: str) -> List[Dict[str, Any]]:
        """Extract sections from markdown text."""
        sections = []
        lines = content.split('\n')
        current_section = None
        section_content = []

        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                if current_section is not None:
                    sections.append({
                        "heading": current_section['heading'],
                        "level": current_section['level'],
                        "content": '\n'.join(section_content).strip(),
                        "start_line": current_section['start_line'],
                        "end_line": i-1
                    })
                heading_text = line.lstrip('#').strip()
                level = len(line) - len(line.lstrip('#'))
                current_section = {'heading': heading_text, 'level': level, 'start_line': i}
                section_content = []
            elif current_section is not None:
                section_content.append(line)

        if current_section is not None:
            sections.append({
                "heading": current_section['heading'],
                "level": current_section['level'],
                "content": '\n'.join(section_content).strip(),
                "start_line": current_section['start_line'],
                "end_line": len(lines)-1
            })

        return sections

    try:
        # Get full content from converter (always extract metadata to get PDF page count)
        result = PdfConverter(
            file_path,
            extract_metadata=True,
            extract_images=extract_images,
            images_output_dir=images_output_dir
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
            metadata = {
                "title": result.title,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                "pdf_page_count": pdf_page_count,
            }
            # Add image metadata if images were extracted
            if extract_images and images:
                metadata["image_count"] = len(images)
                metadata["images_directory"] = images_output_dir or result.metadata.get("images_directory")

        if extract_sections:
            sections = extract_sections_from_text(full_content)

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
async def read_word(
    file_path: str,
    chunk: Optional[int] = 1,
    chunk_size: Optional[int] = 10000,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    extract_sections: Optional[bool] = False,
    extract_metadata: Optional[bool] = False,
    preview_only: Optional[bool] = False,
    preview_lines: Optional[int] = 100,
    session_id: Optional[str] = None,
    return_format: Optional[str] = "text"
) -> Dict[str, Any]:
    """Read and convert a Word document (.docx or .doc) to markdown with formatting preserved.

    USAGE STRATEGY:
    - Preserves document formatting (bold, italic, headings, lists)
    - For large documents: Use preview_only=True first to check structure
    - For structured analysis: Enable extract_sections=True, return_format="json"
    - Handles both .docx (Office 2007+) and .doc (Office 97-2003) formats

    Args:
        file_path: The path to the Word document to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Extract document sections/headings. Use for structured documents. Default: False.
        extract_metadata: Extract file metadata. Use with return_format="json". Default: False.
        preview_only: Return only first N lines for quick assessment. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Reuse for consecutive requests.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    # Apply parameter auto-fix for common naming mistakes
    local_vars = locals().copy()
    fixed_params = fix_tool_arguments("read_word", local_vars)

    # Extract fixed parameters
    file_path = fixed_params.get("file_path", file_path)
    chunk = fixed_params.get("chunk", chunk)
    chunk_size = fixed_params.get("chunk_size", chunk_size)
    offset = fixed_params.get("offset", offset)
    limit = fixed_params.get("limit", limit)
    extract_sections = fixed_params.get("extract_sections", extract_sections)
    extract_metadata = fixed_params.get("extract_metadata", extract_metadata)
    preview_only = fixed_params.get("preview_only", preview_only)
    preview_lines = fixed_params.get("preview_lines", preview_lines)
    session_id = fixed_params.get("session_id", session_id)
    return_format = fixed_params.get("return_format", return_format)

    return await process_document(
        file_path=file_path,
        converter_func=DocxConverter,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
        },
        chunk=chunk,
        chunk_size=chunk_size,
        offset=offset,
        limit=limit,
        extract_sections=extract_sections,
        extract_tables=False,  # Word tables not supported in basic version
        extract_metadata=extract_metadata,
        preview_only=preview_only,
        preview_lines=preview_lines,
        session_id=session_id,
        return_format=return_format
    )


@mcp.tool()
async def read_excel(
    file_path: str,
    chunk: Optional[int] = 1,
    chunk_size: Optional[int] = 10000,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    extract_tables: Optional[bool] = False,
    extract_metadata: Optional[bool] = False,
    preview_only: Optional[bool] = False,
    preview_lines: Optional[int] = 100,
    session_id: Optional[str] = None,
    return_format: Optional[str] = "text"
) -> Dict[str, Any]:
    """Read and convert an Excel file (.xlsx or .xls) to markdown table format.

    USAGE STRATEGY:
    - Converts all worksheets to markdown tables with color formatting preserved
    - For data analysis: Enable extract_tables=True to get structured table data
    - For large spreadsheets: Use preview_only=True to see first few rows
    - Always use return_format="json" with extract_tables=True for programmatic access
    - Handles both .xlsx (Office 2007+) and .xls (Office 97-2003) formats

    Args:
        file_path: The path to the Excel file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_tables: Extract structured table data from each worksheet. Use with return_format="json". Default: False.
        extract_metadata: Extract file metadata. Use with return_format="json". Default: False.
        preview_only: Return only first N lines for quick assessment. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Reuse for consecutive requests.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, tables, etc.
    """
    # Apply parameter auto-fix for common naming mistakes
    local_vars = locals().copy()
    fixed_params = fix_tool_arguments("read_excel", local_vars)

    # Extract fixed parameters
    file_path = fixed_params.get("file_path", file_path)
    chunk = fixed_params.get("chunk", chunk)
    chunk_size = fixed_params.get("chunk_size", chunk_size)
    offset = fixed_params.get("offset", offset)
    limit = fixed_params.get("limit", limit)
    extract_tables = fixed_params.get("extract_tables", extract_tables)
    extract_metadata = fixed_params.get("extract_metadata", extract_metadata)
    preview_only = fixed_params.get("preview_only", preview_only)
    preview_lines = fixed_params.get("preview_lines", preview_lines)
    session_id = fixed_params.get("session_id", session_id)
    return_format = fixed_params.get("return_format", return_format)

    return await process_document(
        file_path=file_path,
        converter_func=XlsxConverter,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_tables": extract_tables,
        },
        chunk=chunk,
        chunk_size=chunk_size,
        offset=offset,
        limit=limit,
        extract_sections=False,  # Excel doesn't have sections
        extract_tables=extract_tables,
        extract_metadata=extract_metadata,
        preview_only=preview_only,
        preview_lines=preview_lines,
        session_id=session_id,
        return_format=return_format
    )


@mcp.tool()
async def read_powerpoint(
    file_path: str,
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
    """Read and convert a PowerPoint presentation (.pptx or .ppt) to markdown.

    USAGE STRATEGY:
    - Extracts all slide content including titles, text, and notes
    - Each slide is formatted as a separate section with heading
    - For quick overview: Use preview_only=True to see first few slides
    - For presentation analysis: Enable extract_sections=True to get slide-by-slide structure
    - Handles both .pptx (Office 2007+) and .ppt (Office 97-2003) formats

    Args:
        file_path: The path to the PowerPoint file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Extract slide sections. Use to get individual slide content. Default: False.
        extract_tables: Extract table information from slides. Default: False.
        extract_metadata: Extract file metadata. Use with return_format="json". Default: False.
        preview_only: Return only first N lines for quick assessment. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Reuse for consecutive requests.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    # Apply parameter auto-fix for common naming mistakes
    local_vars = locals().copy()
    fixed_params = fix_tool_arguments("read_powerpoint", local_vars)

    # Extract fixed parameters
    file_path = fixed_params.get("file_path", file_path)
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

    # Create a wrapper for PptxConverter to support new parameters
    def pptx_converter_wrapper(file_path: str, **kwargs) -> DocumentConverterResult:
        # PptxConverter currently doesn't support extract_metadata, extract_sections, extract_tables
        # We'll call the original function and add metadata extraction if requested
        result = PptxConverter(file_path)

        # If extract_metadata is requested, add file metadata
        if kwargs.get('extract_metadata', False):
            result.metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                "file_extension": os.path.splitext(file_path)[1],
            }

        # If extract_sections is requested, extract sections from markdown
        if kwargs.get('extract_sections', False):
            result.sections = extract_sections_from_markdown(result.text_content)

        # If extract_tables is requested, PowerPoint doesn't have structured table extraction
        # but we can note that in metadata
        if kwargs.get('extract_tables', False):
            result.tables = []  # No table extraction for PowerPoint

        return result

    return await process_document(
        file_path=file_path,
        converter_func=pptx_converter_wrapper,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
            "extract_tables": extract_tables,
        },
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
async def read_html(
    file_path: str,
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
    """Read and convert an HTML file to clean markdown.

    USAGE STRATEGY:
    - Automatically removes scripts, styles, and other non-content elements
    - Preserves semantic structure (headings, lists, tables, links)
    - For web pages: Ideal for extracting readable content from saved HTML
    - For documentation: Enable extract_sections=True to get document structure
    - Security: JavaScript links and data URIs are automatically sanitized

    Args:
        file_path: The path to the HTML file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Extract heading-based sections. Use for structured HTML documents. Default: False.
        extract_tables: Extract table information. Default: False.
        extract_metadata: Extract file metadata. Use with return_format="json". Default: False.
        preview_only: Return only first N lines for quick assessment. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Reuse for consecutive requests.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    return await process_document(
        file_path=file_path,
        converter_func=HtmlConverter,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
        },
        chunk=chunk,
        chunk_size=chunk_size,
        offset=offset,
        limit=limit,
        extract_sections=extract_sections,
        extract_tables=extract_tables,  # HTML may have tables, but not extracted in basic version
        extract_metadata=extract_metadata,
        preview_only=preview_only,
        preview_lines=preview_lines,
        session_id=session_id,
        return_format=return_format
    )


@mcp.tool()
async def read_text(
    file_path: str,
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
    """Read a plain text file (.txt, .md, .py, .sh, etc.) with enhanced features.

    Supports pagination, structured extraction, and session management.

    Args:
        file_path: The path to the text file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Whether to extract document sections/headings. Default: False.
        extract_tables: Whether to extract tables. Default: False.
        extract_metadata: Whether to extract metadata. Default: False.
        preview_only: Whether to return only a preview. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Generated if not provided.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    # Create a wrapper for TextConverter to support new parameters
    def text_converter_wrapper(file_path: str, **kwargs) -> DocumentConverterResult:
        # TextConverter currently doesn't support extract_metadata, extract_sections, extract_tables
        # We'll call the original function and add metadata extraction if requested
        result = TextConverter(file_path)

        # Apply content limit (200,000 characters) like other converters
        result.text_content = apply_content_limit(result.text_content)

        # If extract_metadata is requested, add file metadata
        if kwargs.get('extract_metadata', False):
            result.metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                "file_extension": os.path.splitext(file_path)[1],
            }

        # If extract_sections is requested, extract sections from markdown
        if kwargs.get('extract_sections', False):
            result.sections = extract_sections_from_markdown(result.text_content)

        # Text files don't have tables, but we can note that in metadata
        if kwargs.get('extract_tables', False):
            result.tables = []  # No table extraction for text files

        return result

    return await process_document(
        file_path=file_path,
        converter_func=text_converter_wrapper,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
            "extract_tables": extract_tables,
        },
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
async def read_json(
    file_path: str,
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
    """Read and parse a JSON file with enhanced features.

    Supports pagination, structured extraction, and session management.

    Args:
        file_path: The path to the JSON file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Whether to extract document sections/headings. Default: False.
        extract_tables: Whether to extract tables. Default: False.
        extract_metadata: Whether to extract metadata. Default: False.
        preview_only: Whether to return only a preview. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Generated if not provided.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    # Create a wrapper for JsonConverter to support new parameters
    def json_converter_wrapper(file_path: str, **kwargs) -> DocumentConverterResult:
        # JsonConverter currently doesn't support extract_metadata, extract_sections, extract_tables
        # We'll call the original function and add metadata extraction if requested
        result = JsonConverter(file_path)

        # Apply content limit (200,000 characters) like other converters
        result.text_content = apply_content_limit(result.text_content)

        # If extract_metadata is requested, add file metadata
        if kwargs.get('extract_metadata', False):
            result.metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                "file_extension": os.path.splitext(file_path)[1],
            }

        # If extract_sections is requested, extract sections from markdown
        # JSON files don't have markdown sections, but we can structure by keys
        if kwargs.get('extract_sections', False):
            # For JSON, we can create sections based on top-level keys
            result.sections = []  # No section extraction for JSON in basic version

        # JSON files don't have tables, but we can note that in metadata
        if kwargs.get('extract_tables', False):
            result.tables = []  # No table extraction for JSON files

        return result

    return await process_document(
        file_path=file_path,
        converter_func=json_converter_wrapper,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
            "extract_tables": extract_tables,
        },
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
async def read_csv(
    file_path: str,
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
    """Read a CSV file as markdown table with enhanced features.

    Supports pagination, structured extraction, and session management.

    Args:
        file_path: The path to the CSV file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Whether to extract document sections/headings. Default: False.
        extract_tables: Whether to extract tables. Default: False.
        extract_metadata: Whether to extract metadata. Default: False.
        preview_only: Whether to return only a preview. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Generated if not provided.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    # Create wrapper for CsvConverter
    csv_converter_wrapper = create_simple_converter_wrapper(CsvConverter, "csv")

    return await process_document(
        file_path=file_path,
        converter_func=csv_converter_wrapper,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
            "extract_tables": extract_tables,
        },
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
async def read_yaml(
    file_path: str,
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
    """Read and parse a YAML file (.yaml or .yml) with enhanced features.

    Supports pagination, structured extraction, and session management.

    Args:
        file_path: The path to the YAML file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Whether to extract document sections/headings. Default: False.
        extract_tables: Whether to extract tables. Default: False.
        extract_metadata: Whether to extract metadata. Default: False.
        preview_only: Whether to return only a preview. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Generated if not provided.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    # Create wrapper for YamlConverter
    yaml_converter_wrapper = create_simple_converter_wrapper(YamlConverter, "yaml")

    return await process_document(
        file_path=file_path,
        converter_func=yaml_converter_wrapper,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
            "extract_tables": extract_tables,
        },
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
async def read_zip(
    file_path: str,
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
    """Extract and list contents of a ZIP archive with enhanced features.

    Supports pagination, structured extraction, and session management.

    Args:
        file_path: The path to the ZIP file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Whether to extract document sections/headings. Default: False.
        extract_tables: Whether to extract tables. Default: False.
        extract_metadata: Whether to extract metadata. Default: False.
        preview_only: Whether to return only a preview. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Generated if not provided.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    # Create wrapper for ZipConverter
    def zip_converter_wrapper(file_path: str, **kwargs) -> DocumentConverterResult:
        # ZipConverter already supports **kwargs, but we need to ensure it returns DocumentConverterResult
        result = ZipConverter(file_path, **kwargs)

        # ZipConverter might already apply content limit, but we ensure it
        result.text_content = apply_content_limit(result.text_content)

        # If extract_metadata is requested, add file metadata if not already present
        if kwargs.get('extract_metadata', False) and not result.metadata:
            result.metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                "file_extension": os.path.splitext(file_path)[1],
            }

        # If extract_sections is requested, extract sections from markdown
        if kwargs.get('extract_sections', False) and not result.sections:
            result.sections = extract_sections_from_markdown(result.text_content)

        # If extract_tables is requested
        if kwargs.get('extract_tables', False) and not result.tables:
            result.tables = []  # No table extraction for ZIP files

        return result

    return await process_document(
        file_path=file_path,
        converter_func=zip_converter_wrapper,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
            "extract_tables": extract_tables,
        },
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
async def read_with_markitdown(
    uri: str,
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
    """Convert any file using MarkItDown library (fallback/converter) with enhanced features.

    Supports a wide range of formats including images, audio, video,
    and other document types using MarkItDown's plugins.

    Args:
        uri: The path or URI to the file to convert
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Whether to extract document sections/headings. Default: False.
        extract_tables: Whether to extract tables. Default: False.
        extract_metadata: Whether to extract metadata. Default: False.
        preview_only: Whether to return only a preview. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Generated if not provided.
        return_format: Output format: 'json' (structured) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, etc.
    """
    # Create wrapper for MarkItDownConverter
    def markitdown_converter_wrapper(uri: str, **kwargs) -> DocumentConverterResult:
        # MarkItDownConverter currently doesn't support extract_metadata, extract_sections, extract_tables
        # We'll call the original function and add metadata extraction if requested
        result = MarkItDownConverter(uri)

        # Apply content limit (200,000 characters) like other converters
        result.text_content = apply_content_limit(result.text_content)

        # If extract_metadata is requested, add file metadata
        if kwargs.get('extract_metadata', False):
            result.metadata = {
                "file_path": uri if os.path.exists(uri) else uri,
                "file_size": os.path.getsize(uri) if os.path.exists(uri) else None,
                "file_extension": os.path.splitext(uri)[1] if "." in uri else "",
            }

        # If extract_sections is requested, extract sections from markdown
        if kwargs.get('extract_sections', False):
            result.sections = extract_sections_from_markdown(result.text_content)

        # If extract_tables is requested
        if kwargs.get('extract_tables', False):
            result.tables = []  # No table extraction for MarkItDown

        return result

    return await process_document(
        file_path=uri,  # process_document expects file_path, but uri can be a path
        converter_func=markitdown_converter_wrapper,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
            "extract_tables": extract_tables,
        },
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
