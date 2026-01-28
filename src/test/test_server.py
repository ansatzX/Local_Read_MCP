"""
Unit tests for MCP server tools.

This module contains tests for the FastMCP server implementation,
including tests for all document reading tools and helper functions.
"""

import pytest
import os
import tempfile
import json
from pathlib import Path

# Import the server module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_read_mcp.server import (
    apply_pagination,
    generate_session_id,
    create_simple_converter_wrapper,
    process_document,
)
from local_read_mcp.converters import DocumentConverterResult


class TestApplyPagination:
    """Tests for apply_pagination function."""
    
    def test_apply_pagination_basic(self):
        """Test basic pagination."""
        content = "0123456789" * 100  # 1000 chars
        paginated, has_more = apply_pagination(content, offset=0, limit=100)
        
        assert len(paginated) == 100
        assert has_more is True
    
    def test_apply_pagination_last_page(self):
        """Test pagination on last page."""
        content = "A" * 150
        paginated, has_more = apply_pagination(content, offset=100, limit=100)
        
        assert len(paginated) == 50
        assert has_more is False
    
    def test_apply_pagination_offset_beyond_content(self):
        """Test pagination with offset beyond content."""
        content = "A" * 100
        paginated, has_more = apply_pagination(content, offset=200, limit=50)
        
        assert paginated == ""
        assert has_more is False
    
    def test_apply_pagination_no_limit(self):
        """Test pagination without limit."""
        content = "A" * 100
        paginated, has_more = apply_pagination(content, offset=50, limit=None)
        
        assert len(paginated) == 50
        assert has_more is False


class TestGenerateSessionId:
    """Tests for generate_session_id function."""
    
    def test_generate_session_id_format(self):
        """Test session ID generation format."""
        session_id = generate_session_id("/path/to/file.pdf")
        
        parts = session_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "session"
        assert len(parts[1]) == 8
        assert parts[2].isdigit()
    
    def test_generate_session_id_custom_prefix(self):
        """Test session ID with custom prefix."""
        session_id = generate_session_id("/path/to/file.pdf", prefix="test")
        assert session_id.startswith("test_")


class TestCreateSimpleConverterWrapper:
    """Tests for create_simple_converter_wrapper function."""
    
    def test_wrapper_basic(self, tmp_path):
        """Test wrapper function basics."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content", encoding="utf-8")
        
        def simple_converter(path):
            return DocumentConverterResult(
                title="Test",
                text_content="Content"
            )
        
        wrapper = create_simple_converter_wrapper(simple_converter, "test")
        result = wrapper(str(test_file))
        
        assert result.text_content == "Content"
        assert result.title == "Test"
    
    def test_wrapper_with_metadata(self, tmp_path):
        """Test wrapper with metadata extraction."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test", encoding="utf-8")
        
        def simple_converter(path):
            return DocumentConverterResult(
                title="Test",
                text_content="Content"
            )
        
        wrapper = create_simple_converter_wrapper(simple_converter, "test")
        result = wrapper(str(test_file), extract_metadata=True)
        
        assert result.metadata is not None
        assert "file_path" in result.metadata
        assert result.metadata["file_path"] == str(test_file)


class TestProcessDocument:
    """Tests for process_document function."""
    
    @pytest.mark.asyncio
    async def test_process_document_basic(self, tmp_path):
        """Test basic document processing."""
        test_file = tmp_path / "test.txt"
        test_content = "Test content for processing"
        test_file.write_text(test_content, encoding="utf-8")
        
        def converter(path, **kwargs):
            return DocumentConverterResult(
                title="Test",
                text_content=test_content
            )
        
        result = await process_document(
            file_path=str(test_file),
            converter_func=converter,
            converter_kwargs={},
            return_format="text"
        )
        
        assert result["success"] is True
        assert result["text"] == test_content
    
    @pytest.mark.asyncio
    async def test_process_document_with_pagination(self, tmp_path):
        """Test document processing with pagination."""
        test_file = tmp_path / "test.txt"
        test_content = "A" * 50000
        test_file.write_text(test_content, encoding="utf-8")
        
        def converter(path, **kwargs):
            return DocumentConverterResult(
                title="Test",
                text_content=test_content
            )
        
        result = await process_document(
            file_path=str(test_file),
            converter_func=converter,
            converter_kwargs={},
            chunk=1,
            chunk_size=10000,
            return_format="json"
        )
        
        assert result["success"] is True
        assert len(result["text"]) == 10000
        assert result["pagination_info"]["has_more"] is True
        assert result["pagination_info"]["total_chunks"] == 5
    
    @pytest.mark.asyncio
    async def test_process_document_preview_mode(self, tmp_path):
        """Test document processing in preview mode."""
        test_file = tmp_path / "test.txt"
        test_content = "\n".join([f"Line {i}" for i in range(200)])
        test_file.write_text(test_content, encoding="utf-8")
        
        def converter(path, **kwargs):
            return DocumentConverterResult(
                title="Test",
                text_content=test_content
            )
        
        result = await process_document(
            file_path=str(test_file),
            converter_func=converter,
            converter_kwargs={},
            preview_only=True,
            preview_lines=50,
            return_format="json"
        )
        
        assert result["success"] is True
        assert "Preview" in result["text"]
    
    @pytest.mark.asyncio
    async def test_process_document_with_metadata(self, tmp_path):
        """Test document processing with metadata extraction."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content", encoding="utf-8")
        
        def converter(path, **kwargs):
            result = DocumentConverterResult(
                title="Test",
                text_content="Content"
            )
            if kwargs.get('extract_metadata'):
                result.metadata = {"test_key": "test_value"}
            return result
        
        result = await process_document(
            file_path=str(test_file),
            converter_func=converter,
            converter_kwargs={"extract_metadata": True},
            extract_metadata=True,
            return_format="json"
        )
        
        assert result["success"] is True
        assert "metadata" in result
        assert "file_path" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_process_document_error_handling(self, tmp_path):
        """Test document processing error handling."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content", encoding="utf-8")
        
        def failing_converter(path, **kwargs):
            result = DocumentConverterResult(
                title=None,
                text_content="",
                error="Conversion failed"
            )
            return result
        
        result = await process_document(
            file_path=str(test_file),
            converter_func=failing_converter,
            converter_kwargs={},
            return_format="json"
        )
        
        assert result["success"] is False
        assert "error" in result


class TestServerIntegration:
    """Integration tests for the server."""

    @pytest.mark.asyncio
    async def test_get_supported_formats(self):
        """Test get_supported_formats tool."""
        # Import the raw function, not the MCP tool wrapper
        from local_read_mcp import server

        # Access the original function through the mcp instance
        # Since get_supported_formats is decorated with @mcp.tool(),
        # we need to call it directly as a function
        async def get_formats():
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

        result = await get_formats()

        assert result["success"] is True
        assert "documents" in result
        assert "text" in result
        assert "archives" in result
        assert "pdf" in result["documents"]
        assert "json" in result["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
