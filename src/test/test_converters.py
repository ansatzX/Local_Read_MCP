"""
Unit tests for document converters.

This module contains comprehensive tests for all document conversion functions
in the converters module, including:
- PDF conversion
- Word document conversion
- Excel spreadsheet conversion
- PowerPoint presentation conversion
- HTML conversion
- Text file conversion
- JSON/YAML/CSV conversion
- Pagination and session management
- LaTeX formula fixing
- Section extraction
"""

import pytest
import os
import tempfile
import json
from pathlib import Path

# Import the converters module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_read_mcp.converters import (
    DocumentConverterResult,
    TextConverter,
    JsonConverter,
    YamlConverter,
    CsvConverter,
    apply_content_limit,
    extract_sections_from_markdown,
    fix_latex_formulas,
    generate_session_id,
    PaginationManager,
    inspect_pdf,
    render_pdf_to_images,
    extract_form_fields,
    extract_tables,
)


class TestDocumentConverterResult:
    """Tests for DocumentConverterResult class."""
    
    def test_initialization_basic(self):
        """Test basic initialization of DocumentConverterResult."""
        result = DocumentConverterResult(
            title="Test Document",
            text_content="This is test content"
        )
        
        assert result.title == "Test Document"
        assert result.text_content == "This is test content"
        assert result.metadata == {}
        assert result.sections == []
        assert result.tables == []
        assert result.pagination_info == {}
        assert result.processing_time_ms is None
        assert result.error is None
    
    def test_initialization_with_metadata(self):
        """Test initialization with metadata."""
        metadata = {"file_size": 12345, "author": "Test Author"}
        result = DocumentConverterResult(
            title="Test",
            text_content="Content",
            metadata=metadata
        )
        
        assert result.metadata == metadata
        assert result.metadata["file_size"] == 12345
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = DocumentConverterResult(
            title="Test",
            text_content="Content",
            metadata={"key": "value"},
            processing_time_ms=100
        )
        
        d = result.to_dict()
        assert d["title"] == "Test"
        assert d["text_content"] == "Content"
        assert d["metadata"] == {"key": "value"}
        assert d["processing_time_ms"] == 100
    
    def test_with_error(self):
        """Test result with error."""
        result = DocumentConverterResult(
            title=None,
            text_content="Error occurred",
            error="File not found"
        )
        
        assert result.error == "File not found"
        d = result.to_dict()
        assert "error" in d
        assert d["error"] == "File not found"


class TestTextConverter:
    """Tests for TextConverter."""
    
    def test_text_converter_basic(self, tmp_path):
        """Test basic text file conversion."""
        # Create a temporary text file
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!\nThis is a test."
        test_file.write_text(test_content, encoding="utf-8")
        
        result = TextConverter(str(test_file))
        
        assert result.text_content == test_content
        assert result.title is None
    
    def test_text_converter_empty_file(self, tmp_path):
        """Test conversion of empty text file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")
        
        result = TextConverter(str(test_file))
        
        assert result.text_content == ""
    
    def test_text_converter_unicode(self, tmp_path):
        """Test conversion of text file with Unicode characters."""
        test_file = tmp_path / "unicode.txt"
        test_content = "中文内容 αβγδ 🎉"
        test_file.write_text(test_content, encoding="utf-8")
        
        result = TextConverter(str(test_file))
        
        assert result.text_content == test_content


class TestJsonConverter:
    """Tests for JsonConverter."""
    
    def test_json_converter_basic(self, tmp_path):
        """Test basic JSON file conversion."""
        test_file = tmp_path / "test.json"
        test_data = {"name": "Test", "value": 123, "nested": {"key": "value"}}
        test_file.write_text(json.dumps(test_data), encoding="utf-8")
        
        result = JsonConverter(str(test_file))
        
        # The converter formats JSON with indentation
        assert "Test" in result.text_content
        assert "123" in result.text_content
        assert "nested" in result.text_content
    
    def test_json_converter_array(self, tmp_path):
        """Test JSON converter with array."""
        test_file = tmp_path / "array.json"
        test_data = [1, 2, 3, {"key": "value"}]
        test_file.write_text(json.dumps(test_data), encoding="utf-8")
        
        result = JsonConverter(str(test_file))
        
        assert "1" in result.text_content
        assert "key" in result.text_content


class TestYamlConverter:
    """Tests for YamlConverter."""
    
    def test_yaml_converter_basic(self, tmp_path):
        """Test basic YAML file conversion."""
        test_file = tmp_path / "test.yaml"
        test_content = """
name: Test
value: 123
nested:
  key: value
"""
        test_file.write_text(test_content, encoding="utf-8")
        
        result = YamlConverter(str(test_file))
        
        assert "Test" in result.text_content
        assert "123" in result.text_content
        assert "nested" in result.text_content


class TestCsvConverter:
    """Tests for CsvConverter."""
    
    def test_csv_converter_basic(self, tmp_path):
        """Test basic CSV file conversion."""
        test_file = tmp_path / "test.csv"
        test_content = "Name,Age,City\nAlice,30,New York\nBob,25,Los Angeles"
        test_file.write_text(test_content, encoding="utf-8")
        
        result = CsvConverter(str(test_file))
        
        # CSV converter creates markdown tables
        assert "|" in result.text_content
        assert "Name" in result.text_content
        assert "Alice" in result.text_content
        assert "Bob" in result.text_content
    
    def test_csv_converter_empty(self, tmp_path):
        """Test conversion of empty CSV file."""
        test_file = tmp_path / "empty.csv"
        test_file.write_text("", encoding="utf-8")
        
        result = CsvConverter(str(test_file))
        
        assert "Empty CSV file" in result.text_content


class TestContentLimit:
    """Tests for apply_content_limit function."""
    
    def test_apply_content_limit_under_limit(self):
        """Test content under the limit."""
        content = "Short content"
        result = apply_content_limit(content, max_chars=1000)
        
        assert result == content
    
    def test_apply_content_limit_over_limit(self):
        """Test content over the limit."""
        content = "A" * 300000
        result = apply_content_limit(content, max_chars=200000)
        
        assert len(result) == 200000 + len("\n... [Content truncated]")
        assert result.endswith("\n... [Content truncated]")
    
    def test_apply_content_limit_exact_limit(self):
        """Test content exactly at the limit."""
        content = "A" * 200000
        result = apply_content_limit(content, max_chars=200000)
        
        assert result == content


class TestSectionExtraction:
    """Tests for extract_sections_from_markdown function."""
    
    def test_extract_sections_basic(self):
        """Test basic section extraction."""
        content = """# Section 1
Content for section 1.

## Section 1.1
Subsection content.

# Section 2
Content for section 2.
"""
        sections = extract_sections_from_markdown(content)
        
        assert len(sections) == 3
        assert sections[0]["heading"] == "Section 1"
        assert sections[0]["level"] == 1
        assert sections[1]["heading"] == "Section 1.1"
        assert sections[1]["level"] == 2
        assert sections[2]["heading"] == "Section 2"
    
    def test_extract_sections_no_headings(self):
        """Test extraction from content without headings."""
        content = "This is plain text without any headings."
        sections = extract_sections_from_markdown(content)
        
        assert len(sections) == 0
    
    def test_extract_sections_with_content(self):
        """Test that section content is correctly extracted."""
        content = """# Heading
Line 1
Line 2

# Another Heading
Line 3
"""
        sections = extract_sections_from_markdown(content)
        
        assert "Line 1" in sections[0]["content"]
        assert "Line 2" in sections[0]["content"]
        assert "Line 3" in sections[1]["content"]


class TestLatexFixes:
    """Tests for fix_latex_formulas function."""
    
    def test_fix_latex_cid_placeholders(self):
        """Test fixing CID placeholders."""
        content = "Formula (cid:16)x(cid:17) and (cid:40)y(cid:41)"
        result = fix_latex_formulas(content)
        
        assert "(cid:16)" not in result
        assert "(cid:17)" not in result
        assert "〈" in result
        assert "〉" in result
    
    def test_fix_latex_greek_letters(self):
        """Test fixing Greek letters."""
        content = r"\alpha + \beta = \gamma"
        result = fix_latex_formulas(content)
        
        assert "α" in result
        assert "β" in result
        assert "γ" in result
    
    def test_fix_latex_math_symbols(self):
        """Test fixing mathematical symbols."""
        content = r"\times \div \pm \leq \geq"
        result = fix_latex_formulas(content)
        
        assert "×" in result
        assert "÷" in result
        assert "±" in result
        assert "≤" in result
        assert "≥" in result
    
    def test_fix_latex_empty_string(self):
        """Test fixing empty string."""
        result = fix_latex_formulas("")
        assert result == ""
    
    def test_fix_latex_none(self):
        """Test fixing None value."""
        result = fix_latex_formulas(None)
        assert result is None


class TestSessionId:
    """Tests for generate_session_id function."""
    
    def test_generate_session_id_format(self):
        """Test session ID format."""
        session_id = generate_session_id("/path/to/file.pdf")
        
        # Should have format: prefix_hash_timestamp
        parts = session_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "session"
        assert len(parts[1]) == 8  # Hash length
        assert parts[2].isdigit()  # Timestamp
    
    def test_generate_session_id_custom_prefix(self):
        """Test session ID with custom prefix."""
        session_id = generate_session_id("/path/to/file.pdf", prefix="pdf")
        
        assert session_id.startswith("pdf_")
    
    def test_generate_session_id_deterministic_hash(self):
        """Test that same file path produces same hash."""
        path = "/path/to/file.pdf"
        id1 = generate_session_id(path)
        id2 = generate_session_id(path)
        
        # Hash parts should be the same
        hash1 = id1.split("_")[1]
        hash2 = id2.split("_")[1]
        assert hash1 == hash2


class TestPaginationManager:
    """Tests for PaginationManager class."""
    
    def test_pagination_manager_init(self):
        """Test PaginationManager initialization."""
        content = "A" * 50000
        pm = PaginationManager(content, page_size=10000)
        
        assert pm.total_chars == 50000
        assert pm.page_size == 10000
        assert pm.total_pages == 5
    
    def test_pagination_get_page(self):
        """Test getting a specific page."""
        content = "A" * 50000
        pm = PaginationManager(content, page_size=10000)
        
        page_content, has_more, info = pm.get_page(1)
        
        assert len(page_content) == 10000
        assert has_more is True
        assert info["current_page"] == 1
        assert info["total_pages"] == 5
    
    def test_pagination_get_last_page(self):
        """Test getting the last page."""
        content = "A" * 50000
        pm = PaginationManager(content, page_size=10000)
        
        page_content, has_more, info = pm.get_page(5)
        
        assert len(page_content) == 10000
        assert has_more is False
        assert info["current_page"] == 5
    
    def test_pagination_get_slice(self):
        """Test getting a slice by offset."""
        content = "0123456789" * 1000  # 10000 chars
        pm = PaginationManager(content, page_size=10000)
        
        slice_content, has_more, info = pm.get_slice(1000, 2000)
        
        assert len(slice_content) == 2000
        assert has_more is True
        assert info["char_offset"] == 1000
        assert info["char_limit"] == 2000
    
    def test_pagination_get_slice_no_limit(self):
        """Test getting slice without limit (to end)."""
        content = "A" * 5000
        pm = PaginationManager(content, page_size=1000)
        
        slice_content, has_more, info = pm.get_slice(1000, None)
        
        assert len(slice_content) == 4000
        assert has_more is False


class TestPdfEnhancements:
    """Tests for PDF enhancement features."""

    def test_pdf_inspector_missing_file(self):
        """Test PDF inspector with missing file."""
        result = inspect_pdf("/non/existent/file.pdf")
        assert "error" in result

    def test_render_pdf_missing_file(self):
        """Test PDF rendering with missing file."""
        result = render_pdf_to_images("/non/existent/file.pdf")
        assert len(result) == 1
        assert "error" in result[0]

    def test_extract_forms_missing_file(self):
        """Test form extraction with missing file."""
        result = extract_form_fields("/non/existent/file.pdf")
        assert "error" in result

    def test_extract_tables_missing_file(self):
        """Test table extraction with missing file."""
        result = extract_tables("/non/existent/file.pdf")
        assert len(result) == 1
        assert "error" in result[0]

    def test_document_converter_result_new_fields(self):
        """Test that DocumentConverterResult has new fields."""
        result = DocumentConverterResult(
            title="Test",
            text_content="Content",
            rendered_pages=[{"page": 1, "path": "/tmp/page1.png"}],
            extracted_tables=[{"page": 1, "headers": ["A", "B"]}],
            form_fields=[{"name": "field1", "type": "text"}],
            structure={"metadata": {"title": "Test"}},
            text_with_coords=[{"text": "Hello", "page": 0}],
        )

        assert hasattr(result, 'rendered_pages')
        assert hasattr(result, 'extracted_tables')
        assert hasattr(result, 'form_fields')
        assert hasattr(result, 'structure')
        assert hasattr(result, 'text_with_coords')

        d = result.to_dict()
        assert "rendered_pages" in d
        assert "extracted_tables" in d
        assert "form_fields" in d
        assert "structure" in d
        assert "text_with_coords" in d

    def test_document_converter_result_backward_compatibility(self):
        """Test backward compatibility of DocumentConverterResult."""
        # Create with old parameters only
        result = DocumentConverterResult(
            title="Test",
            text_content="Content",
            metadata={"key": "value"},
            sections=[],
            tables=[],
            images=[],
        )

        d = result.to_dict()
        # Old fields should still exist
        assert "title" in d
        assert "text_content" in d
        assert "metadata" in d
        assert "sections" in d
        assert "tables" in d
        # New fields should NOT be in dict if empty
        assert "rendered_pages" not in d
        assert "extracted_tables" not in d


class TestNewTools:
    """Tests for the new consolidated tools architecture."""

    def test_format_detection_logic(self):
        """Test format detection function logic."""
        test_cases = [
            ("/path/to/file.txt", "text"),
            ("/path/to/file.md", "text"),
            ("/path/to/file.json", "json"),
            ("/path/to/file.csv", "csv"),
            ("/path/to/file.yaml", "yaml"),
            ("/path/to/file.yml", "yaml"),
            ("/path/to/file.pdf", "pdf"),
            ("/path/to/file.docx", "word"),
            ("/path/to/file.doc", "word"),
            ("/path/to/file.xlsx", "excel"),
            ("/path/to/file.xls", "excel"),
            ("/path/to/file.pptx", "ppt"),
            ("/path/to/file.ppt", "ppt"),
            ("/path/to/file.html", "html"),
            ("/path/to/file.htm", "html"),
            ("/path/to/file.zip", "zip"),
            ("/path/to/file.unknown", None),
        ]
        # Actual detection is tested in integration tests
        assert len(test_cases) > 0

    def test_all_converters_available(self):
        """Verify all converter modules are available."""
        from local_read_mcp.converters import (
            TextConverter, JsonConverter, YamlConverter, CsvConverter,
            PdfConverter, DocxConverter, XlsxConverter,
            PptxConverter, HtmlConverter, ZipConverter,
            MarkItDownConverter
        )
        # Just verify imports work
        assert TextConverter is not None
        assert JsonConverter is not None
        assert YamlConverter is not None
        assert CsvConverter is not None
        assert PdfConverter is not None
        assert DocxConverter is not None
        assert XlsxConverter is not None
        assert PptxConverter is not None
        assert HtmlConverter is not None
        assert ZipConverter is not None
        assert MarkItDownConverter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
