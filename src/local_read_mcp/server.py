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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
    Unified document processing with pagination, session management, and structured extraction.

    Args:
        file_path: Path to the file
        converter_func: Converter function to use
        converter_kwargs: Keyword arguments to pass to converter function
        page: Page number for pagination (1-indexed)
        page_size: Number of characters per page
        offset: Character offset (alternative to page)
        limit: Character limit (alternative to page_size)
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
            if page is None or page < 1:
                page = 1
            if page_size is None or page_size < 1:
                page_size = 10000
            char_offset = (page - 1) * page_size
            char_limit = page_size

        # Apply pagination
        paginated_content, has_more = apply_pagination(full_content, char_offset, char_limit)

        # Apply preview if requested
        if preview_only:
            lines = paginated_content.split('\n')
            if len(lines) > preview_lines:
                paginated_content = '\n'.join(lines[:preview_lines]) + f"\n\n... [Preview: showing first {preview_lines} of {len(lines)} lines]"
                has_more = False

        # Generate session ID if not provided
        if not session_id:
            session_id = generate_session_id(file_path, prefix=converter_func.__name__.lower().replace('converter', ''))

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Prepare result based on return format
        if return_format.lower() == "json":
            return {
                "success": True,
                "text": paginated_content,
                "content": paginated_content,  # Keep for compatibility
                "title": result.title,
                "metadata": {**result.metadata, **{
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                }} if extract_metadata else {},
                "sections": result.sections if extract_sections else [],
                "tables": result.tables if extract_tables else [],
                "pagination_info": {
                    "total_pages": max(1, (len(full_content) + char_limit - 1) // char_limit) if char_limit else 1,
                    "current_page": page if offset is None else None,
                    "page_size": char_limit,
                    "has_more": has_more,
                    "char_offset": char_offset,
                    "char_limit": char_limit,
                },
                "session_id": session_id,
                "processing_time_ms": processing_time_ms,
            }
        else:
            # Return text format (backward compatible)
            return {
                "success": True,
                "text": paginated_content,
                "content": paginated_content,
                "title": result.title,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
    """Read and convert a PDF file to markdown text with LaTeX formula fixing.

    USAGE STRATEGY:
    - For unknown file size: Start with preview_only=True, preview_lines=100 to assess content
    - For large files (>10k chars): Use pagination with page=1, page_size=10000
    - For detailed analysis: Enable extract_sections=True, extract_metadata=True, return_format="json"
    - For academic papers: LaTeX formulas are automatically fixed (CID placeholders, Greek letters, math symbols)
    - For multi-page reading: Reuse session_id from previous response for better performance

    Args:
        file_path: The path to the PDF file to read
        page: Page number for pagination (1-indexed). Default: 1.
        page_size: Number of characters per page. Default: 10000.
        offset: Character offset (alternative to page). If specified, overrides page.
        limit: Character limit (alternative to page_size). If specified, overrides page_size.
        extract_sections: Extract document sections/headings. Use for structured documents. Default: False.
        extract_tables: Extract table information. Default: False.
        extract_metadata: Extract file metadata (size, path, timestamp). Use with return_format="json". Default: False.
        preview_only: Return only first N lines without full conversion. Use for quick assessment. Default: False.
        preview_lines: Number of lines for preview mode. Default: 100.
        session_id: Session ID for resuming pagination. Reuse for consecutive page requests.
        return_format: Output format: 'json' (structured with metadata/sections) or 'text' (plain). Default: 'text'.

    Returns:
        A dictionary containing the text content or error message.
        If return_format='json', returns enhanced structure with metadata, sections, pagination_info, session_id.
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
        # Get full content from converter
        result = PdfConverter(file_path)
        full_content = result.text_content

        # Apply LaTeX fixes
        fixed_content = fix_latex_formulas(full_content)

        # Calculate pagination parameters
        if offset is not None:
            char_offset = offset
            char_limit = limit
        else:
            if page is None or page < 1:
                page = 1
            if page_size is None or page_size < 1:
                page_size = 10000
            char_offset = (page - 1) * page_size
            char_limit = page_size

        # Apply pagination
        paginated_content, has_more = apply_pagination(fixed_content, char_offset, char_limit)

        # Apply preview if requested
        if preview_only:
            lines = paginated_content.split('\n')
            if len(lines) > preview_lines:
                paginated_content = '\n'.join(lines[:preview_lines]) + f"\n\n... [Preview: showing first {preview_lines} of {len(lines)} lines]"
                has_more = False

        # Extract structured data if requested
        metadata = {}
        sections = []
        tables = []

        if extract_metadata:
            metadata = {
                "title": result.title,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
            }

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
            return {
                "success": True,
                "text": paginated_content,
                "content": paginated_content,  # Keep for compatibility
                "title": result.title,
                "metadata": metadata,
                "sections": sections,
                "tables": tables,
                "pagination_info": {
                    "total_pages": max(1, (len(fixed_content) + char_limit - 1) // char_limit) if char_limit else 1,
                    "current_page": page if offset is None else None,
                    "page_size": char_limit,
                    "has_more": has_more,
                    "char_offset": char_offset,
                    "char_limit": char_limit,
                },
                "session_id": session_id,
                "processing_time_ms": processing_time_ms,
            }
        else:
            # Return text format (backward compatible)
            return {
                "success": True,
                "text": paginated_content,
                "content": paginated_content,
                "title": result.title,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
    return await process_document(
        file_path=file_path,
        converter_func=DocxConverter,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_sections": extract_sections,
        },
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
    return await process_document(
        file_path=file_path,
        converter_func=XlsxConverter,
        converter_kwargs={
            "extract_metadata": extract_metadata,
            "extract_tables": extract_tables,
        },
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
        page=page,
        page_size=page_size,
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
    page: Optional[int] = 1,
    page_size: Optional[int] = 10000,
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
        page=page,
        page_size=page_size,
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
