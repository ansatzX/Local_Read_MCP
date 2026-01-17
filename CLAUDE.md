# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Local Read MCP Server** processes various file formats locally without external APIs. Converts binary documents (PDF, Word, Excel, PowerPoint, etc.) to readable markdown/text.

## Project Structure

```
local_read_mcp/
├── src/local_read_mcp/
│   ├── __init__.py          # Package initialization
│   ├── server.py            # MCP server implementation (406 lines)
│   └── converters.py        # Document converter classes (842 lines)
├── pyproject.toml           # Python project configuration (requires uv)
├── README.md                # Documentation
├── example_usage.py         # Client usage example
└── LICENSE                  # MIT License
```

## Development Commands

### Dependency Management (uv required)
```bash
# Install in development mode (recommended)
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

**Important**: This project uses uv-specific dependency groups. Using standard `pip` may fail.

### Makefile Commands
```bash
make install       # Production dependencies
make install-dev   # Development dependencies (includes pytest)
make test          # Run all tests
make format        # Format code with ruff
make lint          # Lint code with ruff
make check         # Format and lint (pre-commit check)
make run           # Run MCP server (stdio transport)
make run-http      # Run MCP server (HTTP transport, port 8080)
make clean         # Clean build artifacts
```

### Running Tests
```bash
# Run all tests
pytest src/test/

# Run specific test file
uv run pytest src/test/test_server.py -v

# Run specific test function
uv run pytest src/test/test_server.py::test_server_import -v

# Run with coverage
pytest src/test/ --cov=local_read_mcp --cov-report=term-missing
```

### UV Lock File
The `uv.lock` file ensures reproducible dependency installations. **Do not edit manually** - it updates automatically via `uv pip install` commands.

## Architecture

### MCP Server Implementation
Built using `FastMCP` framework with 12 document processing tools:

1. **Format-specific converters**: `read_pdf()`, `read_word()`, `read_excel()`, `read_powerpoint()`, `read_html()`
2. **Text/data converters**: `read_text()`, `read_json()`, `read_csv()`, `read_yaml()`
3. **Archive processing**: `read_zip()`
4. **Universal fallback**: `read_with_markitdown()`
5. **Utility**: `get_supported_formats()`

### Converter System
- **Modular design**: Each format has a dedicated converter class in `converters.py`
- **Error resilience**: Graceful degradation with fallback to `MarkItDownConverter`
- **Local processing**: No external API dependencies
- **Markdown output**: All formats convert to markdown for LLM compatibility

**Key converter classes**:
- `PdfConverter` - Uses `pdfminer-six` for text extraction
- `DocxConverter` - Uses `mammoth` for Word to HTML → markdown
- `XlsxConverter` - Uses `openpyxl` for Excel to markdown tables
- `PptxConverter` - Uses `python-pptx` for PowerPoint presentations
- `HtmlConverter` - Uses `markdownify` with custom enhancements
- `MarkItDownConverter` - Universal fallback via `markitdown` library

## Claude Code Integration

### Basic Configuration
Add to `~/.config/claude-code-desktop/claude_code_mcp_servers.json`:
```json
{
  "mcpServers": [
    {
      "command": "uv",
      "args": [
        "--directory", "/path/to/Local_Read_MCP",
        "run",
        "--with", "local_read_mcp",
        "python", "-m", "local_read_mcp.server"
      ]
    }
  ]
}
```
Replace `/path/to/Local_Read_MCP` with the actual repository path.

### File Processing Priority
Claude Code automatically uses:
1. **MCP tools** for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.)
2. **Read Tool** for plain text files (.txt, .md, .py, .sh)

**Important**: Binary files are processed locally without using the Read Tool, ensuring proper format conversion rather than raw binary content.

### Optimizing Claude Code Instructions
Add to Claude Code custom instructions:
```
When processing files, always prefer using available MCP tools for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.) instead of the Read tool. This provides better formatted output with proper structure (markdown tables, section headers, etc.). For plain text files (.txt, .md, .py, .sh), the Read Tool is appropriate.
```

## Key Design Decisions

1. **Local-first approach**: Avoids external API dependencies and costs
2. **Markdown as output format**: Ensures compatibility with LLM contexts
3. **Fallback system**: MarkItDown provides extensibility for new formats
4. **Security considerations**: Removes JavaScript links, truncates data URIs in HTML conversion
5. **Transport flexibility**: Supports both stdio (default for MCP) and HTTP transports

## Development Notes

- **Python 3.12+** required (specified in `pyproject.toml`)
- **Async/await pattern**: All tools are async functions
- **Error handling**: Comprehensive try-catch with logging in each tool
- **Test configuration**: Uses `pytest-asyncio` for async test support