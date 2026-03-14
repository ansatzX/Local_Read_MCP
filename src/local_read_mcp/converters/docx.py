import os
import time

from .base import (
    DocumentConverterResult,
    _CustomMarkdownify,
    BeautifulSoup,
    mammoth
)
from .utils import apply_content_limit, extract_sections_from_markdown


def DocxConverter(
    local_path: str,
    extract_metadata: bool = False,
    extract_sections: bool = False
) -> DocumentConverterResult:
    """
    Convert a DOCX file to Markdown format with enhanced features.

    Uses mammoth library to first convert DOCX to HTML, then converts
    the HTML to Markdown.

    Args:
        local_path: Path to the DOCX file to convert.
        extract_metadata: Whether to extract metadata (file size, etc.)
        extract_sections: Whether to extract sections from content

    Returns:
        DocumentConverterResult containing the converted Markdown text and optional metadata/sections.
    """
    if mammoth is None:
        return DocumentConverterResult(
            title=None,
            text_content="[Error: mammoth not installed]",
            error="mammoth not installed"
        )
    if BeautifulSoup is None:
        return DocumentConverterResult(
            title=None,
            text_content="[Error: beautifulsoup4 not installed]",
            error="beautifulsoup4 not installed"
        )
    if _CustomMarkdownify is None:
        return DocumentConverterResult(
            title=None,
            text_content="[Error: markdownify not installed]",
            error="markdownify not installed"
        )

    try:
        with open(local_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html_content = result.value

        # Convert HTML to markdown
        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()

        body_elm = soup.find("body")
        webpage_text = ""
        if body_elm:
            webpage_text = _CustomMarkdownify().convert_soup(body_elm)
        else:
            webpage_text = _CustomMarkdownify().convert_soup(soup)

        assert isinstance(webpage_text, str)

        # Apply content limit
        webpage_text = apply_content_limit(webpage_text)

        # Prepare metadata
        metadata = {}
        sections = []

        if extract_metadata:
            metadata = {
                "file_path": local_path,
                "file_size": os.path.getsize(local_path) if os.path.exists(local_path) else None,
                "file_extension": os.path.splitext(local_path)[1],
                "conversion_timestamp": time.time()
            }

        if extract_sections:
            sections = extract_sections_from_markdown(webpage_text)

        # Get title from HTML or use filename
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string
        else:
            # Use filename without extension as fallback
            filename = os.path.basename(local_path)
            title = os.path.splitext(filename)[0]

        return DocumentConverterResult(
            title=title,
            text_content=webpage_text,
            metadata=metadata,
            sections=sections,
            tables=[],  # DOCX tables not extracted in basic version
            processing_time_ms=None
        )

    except Exception as e:
        return DocumentConverterResult(
            title=None,
            text_content=f"Error converting DOCX: {str(e)}",
            error=str(e)
        )
