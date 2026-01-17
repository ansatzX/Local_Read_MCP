# Local Read MCP Server

Model Context Protocol server for local document processing without external API dependencies.

## Features

- **Local Processing**: No cloud services or API keys required
- **Multi-format Support**: PDF, Word, Excel, PowerPoint, HTML, Text, JSON, CSV, YAML, ZIP
- **Markdown Output**: All formats convert to readable markdown/text
- **MCP Integration**: Seamless with Claude Code and other MCP clients
- **Advanced Capabilities**: Pagination, structured extraction, LaTeX formula fixing, session management
- **Smart Processing**: Automatic content limiting (200k chars), preview mode, metadata extraction

## Quick Installation

### 1. Clone and Setup

```bash
git clone https://github.com/ansatzX/Local_Read_MCP.git
cd Local_Read_MCP

# Install uv if not already installed: https://github.com/astral-sh/uv

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### 2. Configure Claude Code

Edit `~/`claude `/claude-code-desktop/settings.json` .

**Note**: Claude Code supports both array format (`"mcpServers": [{ ... }]`) and object format (`"mcpServers": { "Local_Read": { ... } }`).

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

**Replace `/path/to/Local_Read_MCP` with your actual repository path.**

### 3. Restart Claude Code

Restart Claude Code to load the MCP server.

## Available Tools

All tools support enhanced parameters for pagination, structured extraction, and content control.

| Tool                      | Description                       | Formats             |
| ------------------------- | --------------------------------- | ------------------- |
| `read_pdf`              | PDF to markdown with LaTeX fixing | .pdf                |
| `read_word`             | Word to markdown                  | .docx, .doc         |
| `read_excel`            | Excel to markdown tables          | .xlsx, .xls         |
| `read_powerpoint`       | PowerPoint to markdown            | .pptx, .ppt         |
| `read_html`             | HTML to markdown                  | .html, .htm         |
| `read_text`             | Plain text files                  | .txt, .md, .py, .sh |
| `read_json`             | Parse and format JSON             | .json               |
| `read_csv`              | CSV to markdown tables            | .csv                |
| `read_yaml`             | Parse YAML                        | .yaml, .yml         |
| `read_zip`              | List ZIP contents                 | .zip                |
| `read_with_markitdown`  | Universal fallback converter      | Many formats        |
| `get_supported_formats` | List all supported formats        | -                   |

### Common Parameters

All read tools accept these optional parameters:

| Parameter            | Type | Default | Description                                |
| -------------------- | ---- | ------- | ------------------------------------------ |
| `page`             | int  | 1       | Page number for pagination (1-indexed)     |
| `page_size`        | int  | 10000   | Characters per page                        |
| `offset`           | int  | None    | Character offset (alternative to page)     |
| `limit`            | int  | None    | Character limit (alternative to page_size) |
| `extract_sections` | bool | False   | Extract document sections/headings         |
| `extract_tables`   | bool | False   | Extract table information                  |
| `extract_metadata` | bool | False   | Extract file metadata                      |
| `preview_only`     | bool | False   | Return preview (first N lines)             |
| `preview_lines`    | int  | 100     | Lines to show in preview mode              |
| `session_id`       | str  | None    | Session ID for pagination tracking         |
| `return_format`    | str  | "text"  | Output format: "text" or "json"            |

## Usage Examples

### Basic Usage

```python
# Simple file reading
read_pdf(file_path="/path/to/document.pdf")

# With pagination
read_pdf(file_path="/path/to/large.pdf", page=1, page_size=10000)

# Using offset/limit
read_pdf(file_path="/path/to/document.pdf", offset=5000, limit=10000)
```

### Advanced Features

```python
# Preview mode (quick overview)
read_pdf(file_path="/path/to/document.pdf", preview_only=True, preview_lines=50)

# Structured extraction
read_pdf(
    file_path="/path/to/document.pdf",
    extract_sections=True,
    extract_metadata=True,
    return_format="json"
)

# Session-based pagination
result = read_pdf(file_path="/path/to/large.pdf", page=1)
session_id = result["session_id"]
next_page = read_pdf(file_path="/path/to/large.pdf", page=2, session_id=session_id)
```

### LaTeX Formula Support

PDF files with LaTeX formulas are automatically processed to convert common symbols:

- CID placeholders: `(cid:16)` to angle brackets
- Greek letters: `\alpha` to α, `\beta` to β, etc.
- Math symbols: `\times` to ×, `\leq` to ≤, etc.

### Content Limiting

Files are automatically limited to 200,000 characters with truncation notice. Use pagination to access full content:

```python
# First 10k characters
read_pdf(file_path="/path/to/huge.pdf", page=1, page_size=10000)

# Next 10k characters
read_pdf(file_path="/path/to/huge.pdf", page=2, page_size=10000)
```

## File Processing Priority

Claude Code automatically uses:

1. **MCP tools** for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.)
2. **Read Tool** for plain text files (.txt, .md, .py, .sh)

## Optimizing Claude Code

Add this to your Claude Code custom instructions:

```
When processing files, always prefer using available MCP tools for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.) instead of the Read tool. This provides better formatted output with proper structure (markdown tables, section headers, etc.). For plain text files (.txt, .md, .py, .sh), the Read Tool is appropriate.
```

Or add to `.claude_code_settings.json`:

```json
{
  "defaultInstructions": "When processing files, always prefer using available MCP tools for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.) instead of the Read tool. This provides better formatted output with proper structure (markdown tables, section headers, etc.). For plain text files (.txt, .md, .py, .sh), the Read Tool is appropriate."
}
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest src/test/ -v

# Run with coverage report
uv run pytest src/test/ --cov=src/local_read_mcp --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Statistics

- Total tests: 45
- Coverage: 29.95%
- All core functions: 100% coverage

### Make Commands

```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint

# Run server (stdio)
make run

# Run server (HTTP)
make run-http
```

## Architecture

### Core Components

- **converters.py**: Document conversion logic for all formats
- **server.py**: MCP server implementation with 12 tools
- **Pagination**: `PaginationManager` class for handling large files
- **Content Control**: Automatic 200k character limit with truncation
- **LaTeX Processing**: Formula fixing for academic documents
- **Session Management**: Track pagination state across requests

### Return Formats

#### Text Format (default)

```json
{
  "success": true,
  "text": "content here",
  "content": "content here",
  "title": "document title"
}
```

#### JSON Format (structured)

```json
{
  "success": true,
  "text": "content here",
  "content": "content here",
  "title": "document title",
  "metadata": {
    "file_path": "/path/to/file",
    "file_size": 12345,
    "conversion_timestamp": 1234567890
  },
  "sections": [
    {
      "heading": "Section 1",
      "level": 1,
      "content": "section content",
      "start_line": 0,
      "end_line": 10
    }
  ],
  "pagination_info": {
    "total_pages": 5,
    "current_page": 1,
    "has_more": true,
    "char_offset": 0,
    "char_limit": 10000
  },
  "session_id": "pdf_abc123_1234567890",
  "processing_time_ms": 150
}
```

## License

MIT License

---

## Claude Code Custom Instructions

This section provides optimized custom instructions for Claude Code in both Chinese and English. These instructions help Claude Code automatically use the correct MCP tools for document processing.

### How to Use

Copy the content from [CUSTOM_INSTRUCTIONS_CN.md](./CUSTOM_INSTRUCTIONS_CN.md) (Chinese) or [CUSTOM_INSTRUCTIONS_EN.md](./CUSTOM_INSTRUCTIONS_EN.md) (English) and paste it in UI.

---

### 中文版本 (Chinese Version)

完整内容请查看: [CUSTOM_INSTRUCTIONS_CN.md](./CUSTOM_INSTRUCTIONS_CN.md)

### English Version

完整内容请查看: [CUSTOM_INSTRUCTIONS_EN.md](./CUSTOM_INSTRUCTIONS_EN.md)
