# Local Read MCP Server

A Model Context Protocol server for processing various file formats **without requiring external API keys or cloud services**.

## Features

- Pure Local Processing - No external API dependencies
- Multiple Format Support - PDF, Word, Excel, PowerPoint, HTML, Text, JSON, CSV, YAML, ZIP
- Markdown Output - All formats converted to readable markdown/text
- MCP Protocol - Easy integration with MCP-compatible clients
- Lightweight - Minimal dependencies

## Installation

```bash
# Install dependencies
pip install mcp fastmcp mammoth markdownify openpyxl pdfminer-six python-pptx markitdown pyyaml

# Or using uv
uv pip install mcp fastmcp mammoth markdownify openpyxl pdfminer-six python-pptx markitdown pyyaml
```

## Usage

### Run as MCP Server

```bash
# Standard I/O transport (default)
python - local_read_mcp.server

# HTTP transport
python - local_read_mcp.server --transport http --port 8080
```

### Available Tools

| Tool Name | Description | Supported Formats |
|-----------|-------------|-------------------|
| `read_pdf` | Convert PDF to markdown | .pdf |
| `read_word` | Convert Word to markdown | .docx, .doc |
| `read_excel` | Convert Excel to markdown tables | .xlsx, .xls |
| `read_powerpoint` | Convert PowerPoint to markdown | .pptx, .ppt |
| `read_html` | Convert HTML to markdown | .html, .htm |
| `read_text` | Read plain text files | .txt, .md, .py, .sh, etc. |
| `read_json` | Parse and format JSON | .json |
| `read_csv` | Convert CSV to markdown tables | .csv |
| `read_yaml` | Parse and format YAML | .yaml, .yml |
| `read_zip` | List and extract ZIP contents | .zip |
| `read_with_markitdown` | Convert using MarkItDown (fallback) | Many formats via plugins |
| `get_supported_formats` | List all supported formats | - |

### Using with Claude Code

#### Configuration

In your Claude Code MCP configuration (`.claude_code_mcp_servers.json` or similar), add to following:

```json
{
  "mcpServers": [
    {
      "command": "uv",
      "args": ["run", "--with", "local_read_mcp", "python", "-m", "local_read_mcp.server"]
    }
  ]
}
```

Or if you have to package installed in your current environment:

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

#### File Processing Priority

Claude Code will automatically use to best available tool:

1. **MCP tools (local_read_mcp)** - Used first for binary file formats:
   - PDF (`read_pdf`)
   - Word (`read_word`)
   - Excel (`read_excel`)
   - PowerPoint (`read_powerpoint`)
   - HTML (`read_html`)
   - JSON (`read_json`)
   - YAML (`read_yaml`)
   - CSV (`read_csv`)
   - ZIP (`read_zip`)

2. **Built-in Read Tool** - Used as fallback for:
   - Plain text files
   - Files not supported by MCP tools

**Important**: This means that all binary files supported by `local_read_mcp` will be processed locally without using to Read Tool, ensuring proper format conversion (PDF to text, Excel to markdown tables, etc.) rather than raw binary content.

#### Setting Default Instructions in Claude Code

To optimize how Claude Code uses to local_read_mcp server, you can add custom instructions in your Claude Code settings:

**Option 1: Via Settings File**

Create or edit `.claude_code_settings.json`:

```json
{
  "defaultInstructions": "When processing files, always prefer using available MCP tools for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.) instead of to Read tool. This provides better formatted output with proper structure (markdown tables, section headers, etc.). For plain text files (.txt, .md, .py, .sh), to Read Tool is appropriate."
}
```

**Option 2: Via Claude Code Settings UI**

1. Open Claude Code settings
2. Navigate to "Custom Instructions"
3. Add to following:

```
When processing files, always prefer using available MCP tools for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.) instead of to Read tool. This provides better formatted output with proper structure (markdown tables, section headers, etc.). For plain text files (.txt, .md, .py, .sh), to Read Tool is appropriate.
```

**Option 3: Per-Session Instructions**

You can also add these instructions in to chat when needed:

```
Please use local_read_mcp tools for processing binary document files. This converts formats like PDF, Excel, Word to readable markdown format automatically.
```

### Example Tool Response

When using `read_excel`:

```json
{
  "success": true,
  "text": "## Sheet1\n\n| Header1 | Header2 |\n|---------|---------|\n| Value1 | Value2 |",
  "content": "## Sheet1\n\n| Header1 | Header2 |\n|---------|---------|\n| Value1 | Value2 |",
  "title": null
}
```

When using `read_pdf`:

```json
{
  "success": true,
  "text": "Document text extracted from PDF...",
  "content": "Document text extracted from PDF...",
  "title": null
}
```

### Example Tool Call (Python)

```python
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp import ClientSession

async def use_local_read_mcp():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "local_read_mcp.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Read a PDF file
            result = await session.call_tool(
                "read_pdf",
                arguments={"file_path": "/path/to/document.pdf"}
            )
            print(result.content[-1].text)

            # Get supported formats
            formats = await session.call_tool(
                "get_supported_formats",
                arguments={}
            )
            print(formats)
```

## Format Details

### PDF
- Uses `pdfminer-six` to extract text
- Preserves document structure
- No external dependencies

### Word (DOCX)
- Uses `mammoth` to convert to HTML, then to markdown
- Preserves headings, paragraphs, lists
- Tables converted to markdown tables

### Excel (XLSX)
- Uses `openpyxl` to read spreadsheet
- Creates markdown tables with proper formatting
- Preserves column widths and cell content

### PowerPoint (PPTX)
- Uses `python-pptx` to read presentations
- Extracts slides, tables, images
- Converts to markdown with slide structure

### HTML
- Uses `markdownify` for HTML to markdown conversion
- Removes scripts and styles
- Properly escapes URLs and URIs

### JSON/YAML
- Parses and formats with proper indentation
- Returns as formatted text

### CSV
- Converts to markdown table format
- First row becomes header
- Aligns columns

### ZIP
- Lists all files in archive
- Reports total file count
- Does not extract files (for security)

### MarkItDown Fallback
- Uses `markitdown` library as universal converter
- Supports many additional formats via plugins
- Can handle images, audio, video files (extracts metadata)

## License

MIT License
