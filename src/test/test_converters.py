"""Tests for the document converters."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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


def test_document_converter_result():
    """Test DocumentConverterResult class."""
    result = DocumentConverterResult(
        title="Test Document",
        text_content="This is test content."
    )

    assert result.title == "Test Document"
    assert result.text_content == "This is test content."

    # Test with None title
    result2 = DocumentConverterResult(text_content="Content only")
    assert result2.title is None
    assert result2.text_content == "Content only"


def test_converter_classes_exist():
    """Test that all converter classes are defined."""
    converters = [
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
        MarkItDownConverter
    ]

    for converter_class in converters:
        assert converter_class is not None
        # Check they have __init__ method
        assert hasattr(converter_class, '__init__')
        # text_content and title are instance attributes, not class attributes
        # We'll test them in instance tests


@patch('os.path.exists')
@patch('builtins.open')
def test_text_converter_simple(mock_open, mock_exists):
    """Test TextConverter with mocked file."""
    mock_exists.return_value = True
    mock_file = MagicMock()
    mock_file.read.return_value = "Test file content"
    mock_open.return_value.__enter__.return_value = mock_file

    converter = TextConverter("/fake/path.txt")

    assert converter.text_content == "Test file content"
    assert converter.title is None


@patch('os.path.exists')
def test_pdf_converter_missing_dependency(mock_exists):
    """Test PdfConverter when pdfminer is not available."""
    pytest.skip("This test requires more sophisticated mocking of file operations")


def test_media_extensions_constants():
    """Test that media extension constants are defined."""
    from local_read_mcp.converters import (
        IMAGE_EXTENSIONS,
        AUDIO_EXTENSIONS,
        VIDEO_EXTENSIONS,
        MEDIA_EXTENSIONS
    )

    assert isinstance(IMAGE_EXTENSIONS, set)
    assert isinstance(AUDIO_EXTENSIONS, set)
    assert isinstance(VIDEO_EXTENSIONS, set)
    assert isinstance(MEDIA_EXTENSIONS, set)

    # Check some expected extensions
    assert 'jpg' in IMAGE_EXTENSIONS
    assert 'mp3' in AUDIO_EXTENSIONS
    assert 'mp4' in VIDEO_EXTENSIONS

    # MEDIA_EXTENSIONS should be union of all
    assert MEDIA_EXTENSIONS == IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])