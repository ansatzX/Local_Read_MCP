# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Local Read MCP Server** is a Model Context Protocol server for processing various file formats locally without requiring external API keys or cloud services. It converts binary document formats (PDF, Word, Excel, PowerPoint, etc.) to readable markdown/text format.

## Project Structure

```
local_read_mcp/
├── src/local_read_mcp/
│   ├── __init__.py          # Package initialization
│   ├── server.py            # MCP server implementation (11096 lines)
│   └── converters.py        # Document converter classes (30445 lines)
├── pyproject.toml           # Python project configuration
├── README.md                # Comprehensive documentation
├── example_usage.py         # Example client usage
└── LICENSE                  # MIT License
```

## Development Commands

### Dependency Management

The project uses `hatchling` as the build system with `pyproject.toml`:

```bash
# Install dependencies (using pip)
pip install mcp fastmcp mammoth markdownify openpyxl pdfminer-six python-pptx markitdown pyyaml

# Or using uv
uv pip install mcp fastmcp mammoth markdownify openpyxl pdfminer-six python-pptx markitdown pyyaml

# Install development dependencies
uv pip install "pytest>=8.4.1" "pytest-asyncio>=1.0.0"
```

### Running the Server

```bash
# Standard I/O transport (default for MCP)
python -m local_read_mcp.server

# HTTP transport
python -m local_read_mcp.server --transport http --port 8080
```

### Testing

Tests are configured to run from `src/test/` directory:

```bash
# Run tests with pytest
pytest src/test/

# Run tests with coverage
pytest src/test/ --cov=local_read_mcp --cov-report=term-missing
```

## Architecture

### MCP Server Implementation

The server is built using `FastMCP` framework and provides 12 tools for document processing:

1. **Format-specific converters**: `read_pdf()`, `read_word()`, `read_excel()`, `read_powerpoint()`, `read_html()`
2. **Text/data converters**: `read_text()`, `read_json()`, `read_csv()`, `read_yaml()`
3. **Archive processing**: `read_zip()`
4. **Universal fallback**: `read_with_markitdown()`
5. **Utility**: `get_supported_formats()`

### Converter System

The converter architecture follows a modular design:

- **Base converter pattern**: Each format has a dedicated converter class in `converters.py`
- **Error resilience**: Graceful degradation with fallback to `MarkItDownConverter`
- **Local processing**: No external API dependencies
- **Markdown output**: All formats convert to markdown for LLM compatibility

**Key converter classes**:
- `PdfConverter` - Uses `pdfminer-six` for text extraction
- `DocxConverter` - Uses `mammoth` for Word to HTML, then to markdown
- `XlsxConverter` - Uses `openpyxl` for Excel to markdown tables
- `PptxConverter` - Uses `python-pptx` for PowerPoint presentations
- `HtmlConverter` - Uses `markdownify` with custom enhancements
- `MarkItDownConverter` - Universal fallback via `markitdown` library

### Integration with Claude Code

**MCP Configuration**:
```json
{
  "mcpServers": [
    {
      "command": "python",
      "args": ["-m", "local_read_mcp.server"]
    }
  ]
}
```

**File Processing Priority**:
1. **MCP tools** for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.)
2. **Built-in Read Tool** as fallback for plain text files

**Important**: Binary files supported by `local_read_mcp` are processed locally without using the Read Tool, ensuring proper format conversion (PDF to text, Excel to markdown tables, etc.) rather than raw binary content.

## Supported Formats

**Documents**: PDF (.pdf), Word (.docx, .doc), Excel (.xlsx, .xls), PowerPoint (.pptx, .ppt), HTML (.html, .htm)
**Text/Data**: Plain text (.txt, .md, .py, .sh, etc.), JSON (.json), YAML (.yaml, .yml), CSV (.csv), TOML (.toml)
**Archives**: ZIP (.zip) - lists contents only
**Fallback**: MarkItDown - supports many additional formats via plugins

## Key Design Decisions

1. **Local-first approach**: Avoids external API dependencies and costs
2. **Markdown as output format**: Ensures compatibility with LLM contexts
3. **Fallback system**: MarkItDown provides extensibility for new formats
4. **Security considerations**: Removes JavaScript links, truncates data URIs in HTML conversion
5. **Transport flexibility**: Supports both stdio (default for MCP) and HTTP transports

## Development Notes

- **Python 3.12+** required (specified in `pyproject.toml`)
- **Async/await pattern**: All tools are async functions for better performance
- **Error handling**: Comprehensive try-catch with logging in each tool
- **No external services**: All processing is local; no API keys required
- **Test configuration**: Uses `pytest-asyncio` for async test support