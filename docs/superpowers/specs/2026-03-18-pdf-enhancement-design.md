# PDF Enhancement Design

**Date:** 2026-03-18
**Project:** Local_Read_MCP
**Status:** Draft

## Overview

Enhance the PDF parsing capabilities of Local_Read_MCP with comprehensive features including rendering, table extraction, form handling, metadata inspection, and coordinate-aware text extraction.

## Approach

**PyMuPDF-Centric with Optional Extras** (Approach 1)

- **Core stack (required):** PyMuPDF (fitz) + pdfminer-six (backward compatibility)
- **Optional extras (installable via [pdf] extra):** pdfplumber (tables), pytesseract (OCR), pillow (images)

## Architecture

```
src/local_read_mcp/
├── converters/
│   ├── pdf.py              # Enhanced PDF converter (backward compatible)
│   ├── pdf_rendering.py    # New: Rendering to images
│   ├── pdf_tables.py       # New: Table extraction (optional)
│   ├── pdf_forms.py        # New: Form field handling
│   └── pdf_inspector.py    # New: Metadata/structure inspection
└── server/
    └── app.py              # Add new MCP tools
```

## New Modules

### pdf_rendering.py
```python
def render_pdf_to_images(
    pdf_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 200,
    page_range: Optional[tuple] = None,
    format: str = "png"
) -> List[Dict]:
    """Render PDF pages to images using PyMuPDF."""
```

### pdf_forms.py
```python
def extract_form_fields(pdf_path: str) -> Dict:
    """Extract all form fields with their values, types, and positions."""

def fill_form_fields(
    pdf_path: str,
    field_values: Dict[str, Any],
    output_path: str,
    flatten: bool = False
) -> bool:
    """Fill form fields and save to new PDF."""
```

### pdf_inspector.py
```python
def inspect_pdf(pdf_path: str) -> Dict:
    """Get comprehensive PDF structure information."""

def get_pdf_metadata(pdf_path: str) -> Dict:
    """Get PDF metadata."""
```

### pdf_tables.py (optional)
```python
def extract_tables(
    pdf_path: str,
    page_range: Optional[tuple] = None,
    method: str = "pdfplumber"
) -> List[Dict]:
    """Extract tables from PDF."""
```

## New MCP Tools

### 1. render_pdf_to_images
Render PDF pages to PNG/JPEG images for visual inspection.

### 2. extract_pdf_tables
Extract tables from PDFs with optional pdfplumber integration.

### 3. extract_pdf_forms
Extract form fields with types, values, and positions.

### 4. inspect_pdf
Get comprehensive PDF metadata and structure.

### 5. extract_pdf_text_with_coords
Extract text with bounding box coordinates.

## Backward Compatibility

- Existing `read_pdf` tool continues to work unchanged
- Existing `PdfConverter()` function signature preserved
- All new tools are additive
- Optional features show helpful installation messages when dependencies missing

## pyproject.toml Updates

```toml
[project.optional-dependencies]
pdf = [
    "pdfplumber>=0.10.0",
    "pytesseract>=0.3.10",
    "pillow>=10.0.0",
]
```

## Implementation Phases

1. **Phase 1:** Enhance pdf.py with PyMuPDF text extraction, add pdf_inspector.py
2. **Phase 2:** Add pdf_rendering.py and render_pdf_to_images tool
3. **Phase 3:** Add pdf_forms.py and form handling tools
4. **Phase 4:** Add pdf_tables.py (optional) with pdfplumber integration
5. **Phase 5:** Add coordinate-aware text extraction

## Success Criteria

- All existing tests pass
- New tools work with helpful error messages when optional deps missing
- Backward compatibility maintained
- Documentation updated
