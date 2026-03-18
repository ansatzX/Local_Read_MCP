# PDF Enhancement Design

**Date:** 2026-03-18
**Project:** Local_Read_MCP
**Status:** Draft

## Overview

Enhance the PDF parsing capabilities of Local_Read_MCP with comprehensive features including rendering, table extraction, form handling, metadata inspection, and coordinate-aware text extraction.

## Approach

**PyMuPDF-Centric with Optional Extras**

- **Core stack (required):** PyMuPDF (fitz) + pdfminer-six (backward compatibility)
- **Optional extras (installable via [pdf] extra):** pdfplumber (tables), pytesseract (OCR), pillow (images)

## Architecture - Unified Tool

**Single enhanced tool:** `read_pdf()` with new parameters to control all features.

```
src/local_read_mcp/
├── converters/
│   ├── pdf.py              # Enhanced PDF converter (backward compatible)
│   ├── pdf_rendering.py    # New: Rendering to images
│   ├── pdf_tables.py       # New: Table extraction (optional)
│   ├── pdf_forms.py        # New: Form field handling
│   └── pdf_inspector.py    # New: Metadata/structure inspection
└── server/
    └── app.py              # Enhanced read_pdf tool with new parameters
```

## Unified API: Enhanced read_pdf()

```python
@mcp.tool()
def read_pdf(
    file_path: str,
    chunk: int = 1,
    chunk_size: int = 10000,
    extract_metadata: bool = False,      # Existing parameter
    extract_sections: bool = False,      # Existing parameter
    extract_images: bool = False,        # Existing parameter
    # New parameters
    render_images: bool = False,          # Render pages to images
    render_dpi: int = 200,               # Render DPI
    render_format: str = "png",          # Render format (png/jpeg)
    extract_tables: bool = False,         # Extract tables
    extract_forms: bool = False,          # Extract form fields
    inspect_struct: bool = False,         # Get full structure/metadata
    include_coords: bool = False,         # Include text coordinates
    images_output_dir: Optional[str] = None,
    return_format: str = "json"
) -> Dict:
    """
    Read PDF file with comprehensive feature support.

    New features enabled via parameters:
    - render_images: Render pages to images for visual inspection
    - extract_tables: Extract tables (requires pdfplumber)
    - extract_forms: Extract form fields with types/values/positions
    - inspect_struct: Get complete structure/metadata/outline/fonts
    - include_coords: Include bounding box coordinates with text
    """
```

## Return Format

```json
{
  "title": "Document Title",
  "text_content": "...",
  "metadata": {...},
  "sections": [...],
  "tables": [],
  "images": [...],
  // New optional fields
  "rendered_pages": [
    {
      "page": 1,
      "path": "/tmp/pdf_images/page001.png",
      "width": 1654,
      "height": 2339,
      "dpi": 200
    }
  ],
  "extracted_tables": [
    {
      "page": 1,
      "table_index": 0,
      "headers": ["Column 1", "Column 2"],
      "rows": [["A", "B"], ["C", "D"]],
      "markdown": "| Column 1 | Column 2 |\n|----------|----------|\n| A        | B        |"
    }
  ],
  "form_fields": [
    {
      "name": "first_name",
      "type": "text",
      "value": "",
      "rect": [x0, y0, x1, y1],
      "page": 0
    }
  ],
  "structure": {
    "metadata": {
      "title": "...",
      "author": "...",
      "subject": "...",
      "creation_date": "...",
      "modification_date": "..."
    },
    "page_count": 5,
    "outline": [
      {"title": "Section 1", "page": 0, "level": 1}
    ],
    "fonts": [...],
    "has_acroform": false,
    "is_encrypted": false
  },
  "text_with_coords": [
    {
      "text": "Hello",
      "page": 0,
      "rect": [x0, y0, x1, y1],
      "font": "Times-Roman",
      "size": 12
    }
  ]
}
```

## Internal Modules

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
```

### pdf_inspector.py
```python
def inspect_pdf(pdf_path: str) -> Dict:
    """Get comprehensive PDF structure information."""
```

### pdf_tables.py (optional)
```python
def extract_tables(
    pdf_path: str,
    page_range: Optional[tuple] = None
) -> List[Dict]:
    """Extract tables from PDF using pdfplumber."""
```

## Backward Compatibility

- **Guaranteed:** All existing parameters and behavior unchanged
- **Additive only:** New parameters default to False/off
- **Graceful degradation:** Optional features show helpful error messages:
  ```
  "pdfplumber not installed. Install with: uv pip install 'local_read_mcp[pdf]'"
  ```

## pyproject.toml Updates

```toml
[project.optional-dependencies]
pdf = [
    "pdfplumber>=0.10.0",
    "pillow>=10.0.0",
]
```

## Implementation Phases

1. **Phase 1:** Enhance pdf.py with PyMuPDF, add pdf_inspector.py + `inspect_struct` param
2. **Phase 2:** Add pdf_rendering.py + `render_images` params
3. **Phase 3:** Add pdf_forms.py + `extract_forms` param
4. **Phase 4:** Add pdf_tables.py (optional) + `extract_tables` param
5. **Phase 5:** Add `include_coords` param for coordinate-aware text

## Success Criteria

- All existing tests pass
- Existing `read_pdf` calls continue to work without changes
- New parameters work as documented
- Helpful error messages for missing optional dependencies
- Documentation updated
