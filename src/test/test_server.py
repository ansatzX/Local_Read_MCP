"""
Unit tests for MCP server tools.

This module contains tests for the FastMCP server implementation.
"""


# Import the server module
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from local_read_mcp.converters import DocumentConverterResult, generate_session_id
from local_read_mcp.server import app as server_app
from local_read_mcp.server import (
    apply_pagination,
    create_simple_converter_wrapper,
    process_document,
)


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


class TestProcessBinaryFileExtractImagesDefault:
    """Tests for extract_images default behavior in process_binary_file."""

    @pytest.mark.asyncio
    async def test_auto_enable_extract_images_when_vision_enabled(self, monkeypatch, tmp_path):
        """When not specified and vision is enabled, processing should extract images for PDFs."""
        called = {}
        test_file = tmp_path / "sample.pdf"
        test_file.write_text("fake pdf", encoding="utf-8")

        class FakeBackend:
            name = "Fake"
            warning = None

            def supports_format(self, format_name):
                return True

            def process(self, file_path, format_name, **kwargs):
                called.update(kwargs)
                return {
                    "source": {"path": str(file_path), "format": format_name, "page_count": 1},
                    "metadata": {},
                    "blocks": {
                        "block_00000000": {
                            "type": "text",
                            "page": 1,
                            "bbox": [0, 0, 612, 792],
                            "confidence": 0.9,
                            "content": "content",
                        }
                    },
                    "reading_order": ["block_00000000"],
                }

        class FakeRegistry:
            def select_best(self, format_name=None):
                return FakeBackend()

            def get(self, backend_type):
                return FakeBackend()

        monkeypatch.setattr(server_app, "get_registry", lambda: FakeRegistry())
        monkeypatch.setattr(server_app, "VISION_ENABLED", True)

        result = await server_app.process_binary_file.fn(
            file_path=str(test_file),
            format="pdf",
        )

        assert result["success"] is True
        assert called["extract_images"] is True

    @pytest.mark.asyncio
    async def test_default_extract_images_false_when_vision_disabled(self, monkeypatch, tmp_path):
        """When vision is disabled and not specified, processing should not extract images."""
        called = {}
        test_file = tmp_path / "sample.pdf"
        test_file.write_text("fake pdf", encoding="utf-8")

        class FakeBackend:
            name = "Fake"
            warning = None

            def supports_format(self, format_name):
                return True

            def process(self, file_path, format_name, **kwargs):
                called.update(kwargs)
                return {
                    "source": {"path": str(file_path), "format": format_name, "page_count": 1},
                    "metadata": {},
                    "blocks": {
                        "block_00000000": {
                            "type": "text",
                            "page": 1,
                            "bbox": [0, 0, 612, 792],
                            "confidence": 0.9,
                            "content": "content",
                        }
                    },
                    "reading_order": ["block_00000000"],
                }

        class FakeRegistry:
            def select_best(self, format_name=None):
                return FakeBackend()

            def get(self, backend_type):
                return FakeBackend()

        monkeypatch.setattr(server_app, "get_registry", lambda: FakeRegistry())
        monkeypatch.setattr(server_app, "VISION_ENABLED", False)

        result = await server_app.process_binary_file.fn(
            file_path=str(test_file),
            format="pdf",
        )

        assert result["success"] is True
        assert called["extract_images"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
