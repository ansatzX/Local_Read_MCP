import hashlib
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import DocumentConverterResult, BeautifulSoup, _CustomMarkdownify


class PaginationManager:
    """Manages pagination and session state for large documents."""

    def __init__(self, content: str, page_size: int = 10000):
        """
        Initialize pagination manager.

        Args:
            content: The full content to paginate
            page_size: Number of characters per page (default: 10000)
        """
        self.content = content
        self.page_size = page_size
        self.total_chars = len(content)
        self.total_pages = max(1, (self.total_chars + page_size - 1) // page_size)

    def get_page(self, page: int = 1) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Get a specific page of content.

        Args:
            page: Page number (1-indexed)

        Returns:
            Tuple of (page_content, has_more, pagination_info)
        """
        if page < 1:
            page = 1
        if page > self.total_pages:
            page = self.total_pages

        start = (page - 1) * self.page_size
        end = min(start + self.page_size, self.total_chars)

        page_content = self.content[start:end]
        has_more = end < self.total_chars

        pagination_info = {
            "current_page": page,
            "total_pages": self.total_pages,
            "page_size": self.page_size,
            "char_start": start,
            "char_end": end,
            "has_more": has_more,
            "total_chars": self.total_chars
        }

        return page_content, has_more, pagination_info

    def get_slice(self, offset: int, limit: Optional[int] = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Get a slice of content by character offset.

        Args:
            offset: Character offset to start from
            limit: Maximum number of characters to return (None for all remaining)

        Returns:
            Tuple of (slice_content, has_more, pagination_info)
        """
        if offset >= self.total_chars:
            return "", False, {"char_offset": offset, "char_limit": limit, "has_more": False}

        if limit is None:
            end = self.total_chars
            has_more = False
        else:
            end = min(offset + limit, self.total_chars)
            has_more = end < self.total_chars

        slice_content = self.content[offset:end]

        pagination_info = {
            "char_offset": offset,
            "char_limit": limit,
            "char_start": offset,
            "char_end": end,
            "has_more": has_more,
            "total_chars": self.total_chars
        }

        return slice_content, has_more, pagination_info


def generate_session_id(file_path: str, prefix: str = "session") -> str:
    """
    Generate a unique session ID for a file.

    Args:
        file_path: Path to the file
        prefix: Prefix for the session ID

    Returns:
        Unique session ID string
    """
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    timestamp = int(time.time())
    return f"{prefix}_{file_hash}_{timestamp}"


def apply_content_limit(content: str, max_chars: int = 200000) -> str:
    """
    Apply hard limit to content length.

    Args:
        content: Content to limit
        max_chars: Maximum number of characters (default: 200,000)

    Returns:
        Limited content with truncation notice if needed
    """
    if len(content) > max_chars:
        return content[:max_chars] + "\n... [Content truncated]"
    return content


def extract_sections_from_markdown(content: str) -> List[Dict[str, Any]]:
    """
    Extract sections from markdown text based on headings.

    Args:
        content: Markdown content

    Returns:
        List of section dictionaries with heading, level, content, etc.
    """
    sections = []
    lines = content.split('\n')
    current_section = None
    section_content = []

    # Pre-calculate line start positions for O(n) performance
    line_start_positions = [0] * len(lines)
    current_pos = 0
    for i, line in enumerate(lines):
        line_start_positions[i] = current_pos
        current_pos += len(line) + 1  # +1 for newline
    total_chars = current_pos - 1 if lines else 0

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith('#'):
            # Save previous section if exists
            if current_section is not None:
                sections.append({
                    "heading": current_section['heading'],
                    "level": current_section['level'],
                    "content": '\n'.join(section_content).strip(),
                    "start_line": current_section['start_line'],
                    "end_line": i - 1,
                    "char_start": current_section['char_start'],
                    "char_end": line_start_positions[i] - 1 if i > 0 else 0
                })

            # Start new section
            heading_text = line.lstrip('#').strip()
            level = len(line) - len(line.lstrip('#'))
            current_section = {
                'heading': heading_text,
                'level': level,
                'start_line': i,
                'char_start': line_start_positions[i]
            }
            section_content = []
        elif current_section is not None:
            section_content.append(line)

    # Add the last section if exists
    if current_section is not None:
        sections.append({
            "heading": current_section['heading'],
            "level": current_section['level'],
            "content": '\n'.join(section_content).strip(),
            "start_line": current_section['start_line'],
            "end_line": len(lines) - 1,
            "char_start": current_section['char_start'],
            "char_end": total_chars
        })

    return sections


def fix_latex_formulas(content: str) -> str:
    """Fix common LaTeX formula parsing issues from PDF extraction.

    This function replaces common LaTeX parsing artifacts with proper Unicode characters:
    - CID placeholders (cid:XXX) with corresponding characters
    - LaTeX commands like \alpha with Greek letters (α)
    - Mathematical symbols like \times with Unicode (×)
    - Simplifies superscripts and subscripts notation

    Args:
        content: Content string with LaTeX formulas that need fixing

    Returns:
        Content with fixed LaTeX formulas converted to Unicode

    Example:
        >>> content = "Formula (cid:16)x(cid:17)"
        >>> result = fix_latex_formulas(content)
        >>> "〈x〉" in result
        True

    Note:
        This is a best-effort conversion. Some complex LaTeX formulas
        may not be fully converted.
    """
    if not content:
        return content

    # Fix (cid:XXX) placeholders - using simple replace instead of regex
    cid_map = {
        '(cid:16)': '〈',
        '(cid:17)': '〉',
        '(cid:40)': '(',
        '(cid:41)': ')',
        '(cid:91)': '[',
        '(cid:93)': ']',
        '(cid:123)': '{',
        '(cid:125)': '}',
        '(cid:60)': '<',
        '(cid:62)': '>',
        '(cid:34)': '"',
        '(cid:39)': "'",
        '(cid:44)': ',',
        '(cid:46)': '.',
        '(cid:58)': ':',
        '(cid:59)': ';',
        '(cid:61)': '=',
        '(cid:43)': '+',
        '(cid:45)': '-',
        '(cid:42)': '*',
        '(cid:47)': '/',
        '(cid:92)': '\\',
        '(cid:124)': '|',
    }
    for pattern, replacement in cid_map.items():
        content = content.replace(pattern, replacement)

    # Fix Greek letters - use replace instead of re.sub
    greek_map = {
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\gamma': 'γ',
        r'\delta': 'δ',
        r'\epsilon': 'ε',
        r'\zeta': 'ζ',
        r'\eta': 'η',
        r'\theta': 'θ',
        r'\iota': 'ι',
        r'\kappa': 'κ',
        r'\lambda': 'λ',
        r'\mu': 'μ',
        r'\nu': 'ν',
        r'\xi': 'ξ',
        r'\pi': 'π',
        r'\rho': 'ρ',
        r'\sigma': 'σ',
        r'\tau': 'τ',
        r'\upsilon': 'υ',
        r'\phi': 'φ',
        r'\chi': 'χ',
        r'\psi': 'ψ',
        r'\omega': 'ω',
        r'\Alpha': 'Α',
        r'\Beta': 'Β',
        r'\Gamma': 'Γ',
        r'\Delta': 'Δ',
        r'\Epsilon': 'Ε',
        r'\Zeta': 'Ζ',
        r'\Eta': 'Η',
        r'\Theta': 'Θ',
        r'\Iota': 'Ι',
        r'\Kappa': 'Κ',
        r'\Lambda': 'Λ',
        r'\Mu': 'Μ',
        r'\Nu': 'Ν',
        r'\Xi': 'Ξ',
        r'\Pi': 'Π',
        r'\Rho': 'Ρ',
        r'\Sigma': 'Σ',
        r'\Tau': 'Τ',
        r'\Upsilon': 'Υ',
        r'\Phi': 'Φ',
        r'\Chi': 'Χ',
        r'\Psi': 'Ψ',
        r'\Omega': 'Ω',
    }
    for latex_cmd, unicode_char in greek_map.items():
        content = content.replace(latex_cmd, unicode_char)

    # Fix mathematical symbols
    math_map = {
        r'\times': '×',
        r'\div': '÷',
        r'\pm': '±',
        r'\mp': '∓',
        r'\leq': '≤',
        r'\geq': '≥',
        r'\neq': '≠',
        r'\approx': '≈',
        r'\equiv': '≡',
        r'\propto': '∝',
        r'\infty': '∞',
        r'\partial': '∂',
        r'\nabla': '∇',
        r'\cdot': '·',
        r'\cdots': '⋯',
        r'\vdots': '⋮',
        r'\ddots': '⋱',
        r'\int': '∫',
        r'\sum': '∑',
        r'\prod': '∏',
        r'\cup': '∪',
        r'\cap': '∩',
        r'\in': '∈',
        r'\notin': '∉',
        r'\subset': '⊂',
        r'\supset': '⊃',
        r'\subseteq': '⊆',
        r'\supseteq': '⊇',
        r'\emptyset': '∅',
        r'\forall': '∀',
        r'\exists': '∃',
        r'\neg': '¬',
        r'\wedge': '∧',
        r'\vee': '∨',
        r'\rightarrow': '→',
        r'\leftarrow': '←',
        r'\Rightarrow': '⇒',
        r'\Leftarrow': '⇐',
        r'\Leftrightarrow': '⇔',
    }
    for latex_cmd, unicode_char in math_map.items():
        content = content.replace(latex_cmd, unicode_char)

    # Fix superscripts and subscripts
    content = re.sub(r'\^\{(\d+)\}', r'^\1', content)  # ^{2} → ^2
    content = re.sub(r'_\{(\d+)\}', r'_\1', content)  # _{2} → _2
    content = re.sub(r'\^\{([a-zA-Z])\}', r'^\1', content)  # ^{x} → ^x
    content = re.sub(r'_\{([a-zA-Z])\}', r'_\1', content)  # _{x} → _x

    return content


def html_to_markdown_result(
    html_content: str,
    file_path: str,
    extract_metadata: bool = False,
    extract_sections: bool = False,
    extract_tables: bool = False
) -> DocumentConverterResult:
    """Convert HTML content to a full DocumentConverterResult with enhanced features.

    This shared function handles the common logic for both HtmlConverter and DocxConverter:
    - BeautifulSoup parsing and script/style removal
    - Markdown conversion via _CustomMarkdownify
    - Content limiting
    - Metadata extraction
    - Sections extraction
    - Title fallback from filename

    Args:
        html_content: Raw HTML content string
        file_path: Path to the original file (for metadata and title fallback)
        extract_metadata: Whether to extract file metadata
        extract_sections: Whether to extract sections from markdown
        extract_tables: Whether to extract tables (not implemented yet)

    Returns:
        DocumentConverterResult with converted content and optional metadata/sections
    """
    from .base import BeautifulSoup, _CustomMarkdownify

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

    # Parse HTML and remove scripts/styles
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()

    # Convert to markdown
    body_elm = soup.find("body")
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
        file_size = None
        try:
            file_size = os.path.getsize(file_path)
        except (OSError, Exception):
            pass
        metadata = {
            "file_path": file_path,
            "file_size": file_size,
            "file_extension": os.path.splitext(file_path)[1],
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
        filename = os.path.basename(file_path)
        title = os.path.splitext(filename)[0]

    return DocumentConverterResult(
        title=title,
        text_content=webpage_text,
        metadata=metadata,
        sections=sections,
        tables=[],  # Tables not extracted in basic version
        processing_time_ms=None
    )
