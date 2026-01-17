## Local Read MCP Document Processing Strategy

When processing document files, strictly follow these rules:

#### 1. Tool Selection Priority

**Use MCP tools for binary files**:
- PDF files -> `read_pdf` tool (auto-fix LaTeX formulas, support math symbols)
- Word documents (.docx/.doc) -> `read_word` tool (preserve formatting, convert to markdown)
- Excel spreadsheets (.xlsx/.xls) -> `read_excel` tool (convert to markdown tables, preserve colors)
- PowerPoint (.pptx/.ppt) -> `read_powerpoint` tool (extract slide content)
- HTML files -> `read_html` tool (auto-clean scripts and styles)
- ZIP archives -> `read_zip` tool (auto-extract and process contents)
- JSON files -> `read_json` tool (formatted output)
- CSV files -> `read_csv` tool (convert to markdown tables)
- YAML files -> `read_yaml` tool (parse YAML structure)

**Avoid using Read tool**:
- **[Wrong]** Do not use Read tool for binary files (PDF, Word, Excel, PPT, etc.)
- **[Wrong]** Read tool returns garbled or unparseable content
- **[Correct]** Read tool is only for plain text files (.txt, .md, .py, .sh, .log, etc.)

#### 2. Large File Processing Strategy

For large files (estimated over 10,000 characters), use a step-by-step strategy:

**Step 1 - Preview Mode**:
```python
# Quick preview of first 100 lines to assess file nature
read_pdf(
    file_path="large_file.pdf",
    preview_only=True,
    preview_lines=100
)
```

**Step 2 - Get Metadata**:
```python
# Understand file size and structure
read_pdf(
    file_path="large_file.pdf",
    extract_metadata=True,
    return_format="json"
)
```

**Step 3 - Paginated Processing**:
```python
# Read page by page to avoid timeout
read_pdf(
    file_path="large_file.pdf",
    page=1,              # Page number starts from 1
    page_size=10000,     # 10000 characters per page
    return_format="json"
)
```

**Step 4 - Precise Extraction** (optional):
```python
# Use offset and limit to extract specific content position
read_pdf(
    file_path="large_file.pdf",
    offset=5000,         # Start from character 5000
    limit=3000,          # Read 3000 characters
    return_format="json"
)
```

#### 3. Structured Extraction Features

When deep analysis is needed, enable structured extraction:

**Extract Section Structure**:
```python
read_pdf(
    file_path="document.pdf",
    extract_sections=True,    # Extract all headings and sections
    return_format="json"      # Get sections array
)
```

**Extract Table Information** (Excel only):
```python
read_excel(
    file_path="data.xlsx",
    extract_tables=True,      # Extract table info from each worksheet
    return_format="json"
)
```

**Extract File Metadata**:
```python
read_pdf(
    file_path="document.pdf",
    extract_metadata=True,    # Get file size, path, timestamp, etc.
    return_format="json"
)
```

#### 4. Academic Paper Special Processing

When processing PDFs with math formulas, `read_pdf` auto-fixes LaTeX symbols:
- CID placeholders `(cid:16)` -> left angle bracket
- LaTeX commands `\alpha` -> Greek letter alpha
- Math symbols `\sum` -> summation symbol
- Math symbols `\int` -> integral symbol

This is crucial for accurate understanding of scientific literature and technical documents.

#### 5. Session Management

When processing the same file multiple times, reuse session_id for better performance:

```python
# First request, get session_id
result1 = read_pdf(
    file_path="document.pdf",
    page=1,
    return_format="json"
)
session_id = result1["session_id"]

# Subsequent requests reuse session
result2 = read_pdf(
    file_path="document.pdf",
    page=2,
    session_id=session_id,    # Reuse session
    return_format="json"
)
```

#### 6. Return Format Selection

**text format** (default):
- Use for simple tasks
- Returns only text content and title
- Best compatibility

**json format** (recommended for complex tasks):
- Returns complete structured data
- Includes metadata, sections, tables, pagination info
- Facilitates programmatic processing and deep analysis

```python
# Simple task
result = read_pdf(file_path="doc.pdf")  # Default text format

# Complex analysis
result = read_pdf(
    file_path="doc.pdf",
    extract_sections=True,
    extract_metadata=True,
    return_format="json"  # Get complete structure
)
```

#### 7. Performance Optimization Checklist

Before processing documents, self-check:

- Using the correct MCP tool? (Don't use Read for binary files)
- Preview large files first? (preview_only=True)
- Need structured extraction? (extract_sections/tables/metadata)
- Use JSON format for complex tasks? (return_format="json")
- Consider pagination? (page parameter or offset/limit)
- Reuse session_id for consecutive requests?

#### 8. Common Error Avoidance

**[Wrong Example]**:
```python
# Don't do this!
Read("/path/to/document.pdf")  # Will get garbled output
```

**[Correct Example]**:
```python
# Do this instead
read_pdf("/path/to/document.pdf")
```

**Common Error Comparison**:

<error>
<case>Reading 100MB PDF in one go</case>
<solution>Preview first -> Check size -> Paginate processing</solution>
</error>

<error>
<case>Re-reading entire file every time</case>
<solution>Use session_id and incremental pagination</solution>
</error>

<error>
<case>Getting plain text only, losing sections and table info</case>
<solution>Enable extract_* parameters, use return_format="json"</solution>
</error>

---

**Follow these rules to ensure efficient and accurate document processing, fully leveraging Local Read MCP's capabilities.**
