# Local Read MCP Server

Model Context Protocol server for local document processing without external API dependencies.

## Features

- **Local Processing**: No cloud services or API keys required
- **Multi-format Support**: PDF, Word, Excel, PowerPoint, HTML, Text, JSON, CSV, YAML, ZIP
- **Markdown Output**: All formats convert to readable markdown/text
- **MCP Integration**: Seamless with Claude Code and other MCP clients

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

Edit `~/.config/claude-code-desktop/claude_code_mcp_servers.json` (macOS/Linux) or `%APPDATA%\claude-code-desktop\claude_code_mcp_servers.json` (Windows).

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

| Tool                      | Description                  | Formats             |
| ------------------------- | ---------------------------- | ------------------- |
| `read_pdf`              | PDF to markdown              | .pdf                |
| `read_word`             | Word to markdown             | .docx, .doc         |
| `read_excel`            | Excel to markdown tables     | .xlsx, .xls         |
| `read_powerpoint`       | PowerPoint to markdown       | .pptx, .ppt         |
| `read_html`             | HTML to markdown             | .html, .htm         |
| `read_text`             | Plain text files             | .txt, .md, .py, .sh |
| `read_json`             | Parse and format JSON        | .json               |
| `read_csv`              | CSV to markdown tables       | .csv                |
| `read_yaml`             | Parse YAML                   | .yaml, .yml         |
| `read_zip`              | List ZIP contents            | .zip                |
| `read_with_markitdown`  | Universal fallback converter | Many formats        |
| `get_supported_formats` | List all supported formats   | -                   |

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

## License

MIT License
