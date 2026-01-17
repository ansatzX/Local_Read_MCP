"""Tests for the local_read_mcp server."""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from local_read_mcp.server import mcp


def test_server_import():
    """Test that the server module can be imported."""
    from local_read_mcp.server import mcp
    assert mcp is not None
    assert mcp.name == "local_read_mcp-server"


def test_tools_registered():
    """Test that expected tools are registered."""
    # Check that tool functions exist in the module
    from local_read_mcp.server import (
        read_pdf,
        read_word,
        read_excel,
        read_powerpoint,
        read_html,
        read_text,
        read_json,
        read_csv,
        read_yaml,
        read_zip,
        read_with_markitdown,
        get_supported_formats
    )

    # If imports succeeded, the test passes
    assert True


@pytest.mark.asyncio
async def test_get_supported_formats_tool():
    """Test the get_supported_formats tool."""
    from local_read_mcp.server import get_supported_formats

    # get_supported_formats is decorated by @mcp.tool()
    # The decorator returns a FunctionTool object, not the original function
    assert get_supported_formats is not None

    # Check that it has attributes expected of a FunctionTool
    assert hasattr(get_supported_formats, 'name')
    assert get_supported_formats.name == 'get_supported_formats'
    assert hasattr(get_supported_formats, 'description')


def test_converter_imports():
    """Test that all converter classes can be imported."""
    from local_read_mcp.converters import (
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
        DocumentConverterResult
    )

    # If we get here, imports succeeded
    assert True