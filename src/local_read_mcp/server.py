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
from typing import Dict, Any
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
)

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("local_read_mcp-server")


@mcp.tool()
async def read_pdf(file_path: str) -> Dict[str, Any]:
    """Read and convert a PDF file to markdown text.

    Args:
        file_path: The path to PDF file to

    Returns:
        A dictionary containing to text content or error message.
    """
    try:
        result = PdfConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": result.title,
        }
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read PDF file: {str(e)}",
        }


@mcp.tool()
async def read_word(file_path: str) -> Dict[str, Any]:
    """Read and convert a Word document (.docx or .doc) to markdown.

    Args:
        file_path: The path to the Word document to

    Returns:
        A dictionary containing to markdown content or error message.
    """
    try:
        result = DocxConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": result.title,
        }
    except Exception as e:
        logger.error(f"Error reading Word document: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read Word document: {str(e)}",
        }


@mcp.tool()
async def read_excel(file_path: str) -> Dict[str, Any]:
    """Read and convert an Excel file (.xlsx or .xls) to markdown table format.

    Args:
        file_path: The path to the Excel file to

    Returns:
        A dictionary containing to markdown table content or error message.
    """
    try:
        result = XlsxConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": result.title,
        }
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read Excel file: {str(e)}",
        }


@mcp.tool()
async def read_powerpoint(file_path: str) -> Dict[str, Any]:
    """Read and convert a PowerPoint presentation (.pptx or .ppt) to markdown.

    Args:
        file_path: The path to the PowerPoint file to

    Returns:
        A dictionary containing to markdown content or error message.
    """
    try:
        result = PptxConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": result.title,
        }
    except Exception as e:
        logger.error(f"Error reading PowerPoint file: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read PowerPoint file: {str(e)}",
        }


@mcp.tool()
async def read_html(file_path: str) -> Dict[str, Any]:
    """Read and convert an HTML file to markdown.

    Args:
        file_path: The path to the HTML file to

    Returns:
        A dictionary containing to markdown content or error message.
    """
    try:
        result = HtmlConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": result.title,
        }
    except Exception as e:
        logger.error(f"Error reading HTML file: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read HTML: {str(e)}",
        }


@mcp.tool()
async def read_text(file_path: str) -> Dict[str, Any]:
    """Read a plain text file (.txt, .md, .py, .sh, etc.).

    Args:
        file_path: The path to the text file to

    Returns:
        A dictionary containing to text content.
    """
    try:
        result = TextConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": None,
        }
    except Exception as e:
        logger.error(f"Error reading text file: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read text file: {str(e)}",
        }


@mcp.tool()
async def read_json(file_path: str) -> Dict[str, Any]:
    """Read and parse a JSON file.

    Args:
        file_path: The path to the JSON file to

    Returns:
        A dictionary containing to formatted JSON content or error message.
    """
    try:
        result = JsonConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": None,
        }
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read JSON: {str(e)}",
        }


@mcp.tool()
async def read_csv(file_path: str) -> Dict[str, Any]:
    """Read a CSV file as markdown table.

    Args:
        file_path: The path to the CSV file to

    Returns:
        A dictionary containing to markdown table content or error message.
    """
    try:
        result = CsvConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": None,
        }
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read CSV file: {str(e)}",
        }


@mcp.tool()
async def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read and parse a YAML file (.yaml or .yml).

    Args:
        file_path: The path to the YAML file to

    Returns:
        A dictionary containing to formatted YAML content or error message.
    """
    try:
        result = YamlConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": None,
        }
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read YAML file: {str(e)}",
        }


@mcp.tool()
async def read_zip(file_path: str) -> Dict[str, Any]:
    """Extract and list contents of a ZIP archive.

    Args:
        file_path: The path to the ZIP file to

    Returns:
        A dictionary containing to file listing and contents or error message.
    """
    try:
        result = ZipConverter(file_path)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": "ZIP Archive",
        }
    except Exception as e:
        logger.error(f"Error reading ZIP file: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to read ZIP file: {str(e)}",
        }


@mcp.tool()
async def read_with_markitdown(uri: str) -> Dict[str, Any]:
    """Convert any file using MarkItDown library (fallback/converter).

    Supports a wide range of formats including images, audio, video,
    and other document types using MarkItDown's plugins.

    Args:
        uri: The path or URI to the file to

    Returns:
        A dictionary containing to converted markdown content or error message.
    """
    try:
        result = MarkItDownConverter(uri)
        return {
            "success": True,
            "text": result.text_content,
            "content": result.text_content,
            "title": result.title,
        }
    except Exception as e:
        logger.error(f"Error with MarkItDown: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error: Failed to convert with MarkItDown: {str(e)}",
        }


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
