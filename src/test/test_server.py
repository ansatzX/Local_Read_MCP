"""
Unit tests for MCP server tools.

This module contains tests for the FastMCP server implementation,
including tests for all document reading tools and helper functions.
"""

import pytest
import os
import tempfile
from pathlib import Path

# Import the server module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_read_mcp.server import (
    apply_pagination,
    create_simple_converter_wrapper,
    process_document,
)
from local_read_mcp.server import app as server_app
from local_read_mcp.converters import generate_session_id
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
        """Test get_supported_formats tool returns correct structure."""
        result = await server_app.get_supported_formats.fn()

        assert "text_formats" in result
        assert "binary_formats" in result
        assert "tools" in result
        assert "notes" in result
        assert "migration_guide" not in result
        assert result["tools"]["main"] == ["read_text_file", "read_binary_file"]
        assert result["tools"]["auxiliary"] == ["analyze_image", "get_vision_status", "cleanup_temp_files"]


class TestProcessPdfDocument:
    """Regression tests for process_pdf_document."""

    @pytest.mark.asyncio
    async def test_process_pdf_document_text_response_has_total_chars(self, tmp_path, monkeypatch):
        """Text response should succeed and include total character count."""
        test_file = tmp_path / "sample.pdf"
        test_file.write_text("fake-pdf", encoding="utf-8")

        def fake_pdf_converter(path, **kwargs):
            return DocumentConverterResult(
                title="Sample PDF",
                text_content="PDF content"
            )

        monkeypatch.setattr(server_app, "PdfConverter", fake_pdf_converter)

        result = await server_app.process_pdf_document(
            file_path=str(test_file),
            chunk=1,
            chunk_size=10000,
            offset=None,
            limit=None,
            extract_sections=False,
            extract_tables=False,
            extract_metadata=False,
            extract_images=False,
            render_images=False,
            render_dpi=200,
            render_format="png",
            extract_forms=False,
            inspect_struct=False,
            include_coords=False,
            images_output_dir=None,
            preview_only=False,
            preview_lines=100,
            session_id=None,
            return_format="text",
        )

        assert result["success"] is True
        assert result["total_chars"] == len("PDF content")


class TestConverterWrapperCaching:
    """Tests for converter wrapper caching optimizations."""

    @pytest.mark.asyncio
    async def test_read_text_file_reuses_simple_converter_wrapper(self, monkeypatch, tmp_path):
        """read_text_file should reuse the same simple wrapper across calls."""
        test_file = tmp_path / "sample.txt"
        test_file.write_text("hello", encoding="utf-8")

        wrapper_create_calls = 0
        converter_ids = []

        def fake_wrapper_factory(converter_func, converter_name=""):
            nonlocal wrapper_create_calls
            wrapper_create_calls += 1

            def wrapper(file_path, **kwargs):
                return DocumentConverterResult(title="x", text_content="y")

            return wrapper

        async def fake_process_document(file_path, converter_func, converter_kwargs, **kwargs):
            converter_ids.append(id(converter_func))
            return {"success": True, "text": "ok"}

        if hasattr(server_app, "_SIMPLE_CONVERTER_CACHE"):
            server_app._SIMPLE_CONVERTER_CACHE.clear()

        monkeypatch.setattr(server_app, "create_simple_converter_wrapper", fake_wrapper_factory)
        monkeypatch.setattr(server_app, "process_document", fake_process_document)

        await server_app.read_text_file.fn(file_path=str(test_file), format="text")
        await server_app.read_text_file.fn(file_path=str(test_file), format="text")

        assert wrapper_create_calls == 1
        assert len(converter_ids) == 2
        assert converter_ids[0] == converter_ids[1]


class TestReadBinaryFileExtractImagesDefault:
    """Tests for extract_images default behavior in read_binary_file."""

    @pytest.mark.asyncio
    async def test_auto_enable_extract_images_when_vision_enabled(self, monkeypatch):
        """When not specified and vision is enabled, extract_images should auto-enable."""
        called = {}

        async def fake_process_pdf_document(**kwargs):
            called.update(kwargs)
            return {"success": True}

        monkeypatch.setattr(server_app, "process_pdf_document", fake_process_pdf_document)
        monkeypatch.setattr(server_app, "VISION_ENABLED", True)

        result = await server_app.read_binary_file.fn(file_path="/tmp/a.pdf", format="pdf")

        assert result["success"] is True
        assert called["extract_images"] is True

    @pytest.mark.asyncio
    async def test_explicit_extract_images_false_is_respected(self, monkeypatch):
        """Explicit extract_images=False should not be overridden."""
        called = {}

        async def fake_process_pdf_document(**kwargs):
            called.update(kwargs)
            return {"success": True}

        monkeypatch.setattr(server_app, "process_pdf_document", fake_process_pdf_document)
        monkeypatch.setattr(server_app, "VISION_ENABLED", True)

        result = await server_app.read_binary_file.fn(
            file_path="/tmp/a.pdf",
            format="pdf",
            extract_images=False,
        )

        assert result["success"] is True
        assert called["extract_images"] is False

    @pytest.mark.asyncio
    async def test_default_extract_images_false_when_vision_disabled(self, monkeypatch):
        """When vision is disabled and not specified, extract_images should be False."""
        called = {}

        async def fake_process_pdf_document(**kwargs):
            called.update(kwargs)
            return {"success": True}

        monkeypatch.setattr(server_app, "process_pdf_document", fake_process_pdf_document)
        monkeypatch.setattr(server_app, "VISION_ENABLED", False)

        result = await server_app.read_binary_file.fn(file_path="/tmp/a.pdf", format="pdf")

        assert result["success"] is True
        assert called["extract_images"] is False


class TestParameterExtractionHelpers:
    """Tests for parameter extraction helper functions."""

    def test_extract_common_read_params_basic(self):
        """Helper should extract basic common read parameters."""
        params = {
            "file_path": "/tmp/a.txt",
            "format": "text",
            "chunk": 2,
            "chunk_size": 5000,
            "offset": None,
            "limit": None,
            "extract_sections": True,
            "extract_tables": False,
            "extract_metadata": True,
            "preview_only": False,
            "preview_lines": 50,
            "session_id": "s1",
            "return_format": "json",
        }

        common_params, fixed_params = server_app._extract_common_read_params("read_text_file", params)

        assert common_params["file_path"] == "/tmp/a.txt"
        assert common_params["format"] == "text"
        assert common_params["chunk"] == 2
        assert common_params["chunk_size"] == 5000
        assert common_params["extract_sections"] is True
        assert common_params["extract_metadata"] is True
        assert common_params["return_format"] == "json"
        assert fixed_params["file_path"] == "/tmp/a.txt"

    def test_extract_common_read_params_applies_aliases(self):
        """Helper should apply fix_tool_arguments aliases before extraction."""
        params = {
            "filepath": "/tmp/a.txt",
            "page": 3,
            "page_size": 2000,
            "preview": True,
            "metadata": True,
            "sections": True,
            "tables": True,
            "return_format": "text",
            "format": "text",
            "offset": None,
            "limit": None,
            "preview_lines": 10,
            "session_id": None,
        }

        common_params, _ = server_app._extract_common_read_params("read_text_file", params)

        assert common_params["file_path"] == "/tmp/a.txt"
        assert common_params["chunk"] == 3
        assert common_params["chunk_size"] == 2000
        assert common_params["preview_only"] is True
        assert common_params["extract_metadata"] is True
        assert common_params["extract_sections"] is True
        assert common_params["extract_tables"] is True


class TestFormatExtensionHelpers:
    """Tests for format/extension helper functions."""

    def test_build_supported_format_groups(self):
        """Builder should generate text/binary groups from extension mapping."""
        text_formats, binary_formats = server_app._build_supported_format_groups()

        assert text_formats[0]["name"] == "Plain Text"
        assert text_formats[0]["extensions"] == [".txt", ".md", ".py", ".sh", ".log", ".rst"]
        assert any(item["name"] == "PDF" and item["extensions"] == [".pdf"] for item in binary_formats)
        assert any(item["name"] == "Word" and item["extensions"] == [".docx", ".doc"] for item in binary_formats)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
