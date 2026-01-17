# Local Read MCP Server

Model Context Protocol server for local document processing without external API dependencies.

## Features

- **Local Processing**: No cloud services or API keys required for document conversion
- **Multi-format Support**: PDF, Word, Excel, PowerPoint, HTML, Text, JSON, CSV, YAML, ZIP
- **Markdown Output**: All formats convert to readable markdown/text
- **MCP Integration**: Seamless with Claude Code and other MCP clients
- **Advanced Capabilities**: Pagination, structured extraction, LaTeX formula fixing, session management
- **Smart Processing**: Automatic content limiting (200k chars), preview mode, metadata extraction
- **PDF Image Extraction**: Extract embedded images from PDFs using PyMuPDF
- **Vision Analysis** (Optional): Analyze PDF charts/diagrams with multimodal AI models

## Vision Analysis for PDF Charts

**Note**: Vision analysis is an optional feature for advanced PDF processing.

### When You Need It

The PDF processing system can **extract all images** (both embedded and vector graphics) without any API configuration. However, to **understand the content** of these images (what the charts show, what the diagrams mean, etc.), you need a multimodal vision API.

**Two separate capabilities**:
1. **Image Extraction** (no API needed):
   - Extracts embedded images (photos, screenshots) via `get_images()`
   - Detects vector graphics (charts, diagrams) by analyzing PDF drawing commands
   - Renders vector graphics pages as high-resolution PNG images

2. **Image Content Analysis** (requires vision API):
   - Analyzes what the images show
   - Extracts data from charts
   - Understands diagrams and flowcharts

**What happens with/without API**:
- **Without API**: All images are extracted and saved, but not analyzed. You can view them manually.
- **With API**: All images are extracted AND analyzed by AI to understand their content.

### Affected Use Cases

If your PDFs contain images (embedded or vector graphics) and you want to understand their content automatically:

- Performance comparison charts
- Architecture diagrams
- Flowcharts
- Mathematical plots
- Technical illustrations drawn as vector graphics

Then configuring vision analysis is recommended for complete extraction.

### Configuration (Optional)

1. **Install vision dependencies**:

   ```bash
   uv pip install -e ".[vision]"
   ```
2. **Configure API** (choose one option):

   **Option A: Doubao (ByteDance, recommended)**

   ```bash
   # Create .env file
   cp .env.example .env

   # Edit .env
   OPENAI_API_KEY=your-doubao-api-key
   OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
   OPENAI_VISION_MODEL=doubao-seed-1-8-251228
   ```

   **Option B: OpenAI GPT-4o**

   ```bash
   OPENAI_API_KEY=sk-your-openai-key
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_VISION_MODEL=gpt-4o
   ```

   **Option C: Local Ollama (free, no API key)**

   ```bash
   # Install and start Ollama
   ollama pull llava:13b
   ollama serve

   # Configure in .env
   VISION_DEFAULT_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_VISION_MODEL=llava:13b
   ```

### Behavior Without Configuration

If you don't configure vision analysis:

- Basic PDF reading works normally
- **All images are extracted** (embedded images + vector graphics)
- Image files are saved to disk for manual viewing
- **Image content is not analyzed** - you won't get AI-generated descriptions
- You'll see a warning: `[WARN] Images extracted but content not analyzed`

**This is fine for most use cases** - only configure if you want AI to automatically understand image content.

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

# Optional: Install vision analysis dependencies (for PDF chart analysis)
uv pip install -e ".[vision]"
```

**Optional Dependencies:**

- `vision`: Enables vision analysis for PDF charts using OpenAI/Doubao or Ollama
  - `openai`: For cloud-based vision API (OpenAI GPT-4o or Doubao multimodal models)
  - `aiohttp`: For local Ollama vision models (e.g., LLaVA)

### 2. Configure Claude Code

Edit `~/.claude/settings.json`.

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

| Parameter             | Type | Default | Description                                  |
| --------------------- | ---- | ------- | -------------------------------------------- |
| `page`              | int  | 1       | Page number for pagination (1-indexed)       |
| `page_size`         | int  | 10000   | Characters per page                          |
| `offset`            | int  | None    | Character offset (alternative to page)       |
| `limit`             | int  | None    | Character limit (alternative to page_size)   |
| `extract_sections`  | bool | False   | Extract document sections/headings           |
| `extract_tables`    | bool | False   | Extract table information                    |
| `extract_metadata`  | bool | False   | Extract file metadata                        |
| `extract_images`    | bool | False   | Extract images (PDF only, requires PyMuPDF)  |
| `images_output_dir` | str  | None    | Directory to save images (default: temp dir) |
| `preview_only`      | bool | False   | Return preview (first N lines)               |
| `preview_lines`     | int  | 100     | Lines to show in preview mode                |
| `session_id`        | str  | None    | Session ID for pagination tracking           |
| `return_format`     | str  | "text"  | Output format: "text" or "json"              |

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

### PDF Image Extraction

Extract images from PDF files using PyMuPDF (fitz):

```python
# Extract all images from PDF
read_pdf(
    file_path="/path/to/document.pdf",
    extract_images=True,
    images_output_dir="/tmp/pdf_images",
    extract_metadata=True,
    return_format="json"
)

# Response includes images array with details:
# {
#   "images": [
#     {
#       "page": 0,              # Page number (0-indexed)
#       "index": 0,             # Image index on page
#       "width": 800,           # Image width in pixels
#       "height": 600,          # Image height in pixels
#       "format": "png",        # Image format
#       "size": 123456,         # File size in bytes
#       "saved_path": "/tmp/pdf_images/page000_img00.png"
#     }
#   ],
#   "metadata": {
#     "image_count": 10,
#     "images_directory": "/tmp/pdf_images"
#   }
# }
```

**Requirements**:

- PyMuPDF (fitz): `uv pip install pymupdf`

**Vision Analysis** (Optional):

To analyze extracted images with AI vision models, use the separate `vision_server`:

1. **Configure vision backend** (create `.env` from `.env.example`):

   ```bash
   # Option A: Local Ollama (free, requires GPU)
   VISION_DEFAULT_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_VISION_MODEL=llava:13b

   # Option B: OpenAI GPT-4o (requires API key)
   VISION_DEFAULT_PROVIDER=openai
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_VISION_MODEL=gpt-4o
   ```
2. **Start vision server**:

   ```bash
   python -m local_read_mcp.vision_server
   ```
3. **Configure in Claude Code** (`claude_code_mcp_servers.json`):

   ```json
   {
     "mcpServers": [
       {
         "name": "Local_Read",
         "command": "uv",
         "args": [
           "--directory", "/path/to/Local_Read_MCP",
           "run", "python", "-m", "local_read_mcp.server"
         ]
       },
       {
         "name": "Local_Vision",
         "command": "uv",
         "args": [
           "--directory", "/path/to/Local_Read_MCP",
           "run", "python", "-m", "local_read_mcp.vision_server"
         ]
       }
     ]
   }
   ```
4. **Use in Claude Code**:

   ```
   "请读取 document.pdf 并提取所有图片,然后分析每张图片的内容"
   ```

See `example_pdf_images.py` for complete code examples.

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

This section provides custom instructions for Claude Code in two versions:

1. **Simple Version (Recommended)**: Minimal instruction that relies on MCP tool descriptions for detailed strategies
2. **Detailed Version**: Complete processing strategies in separate files

### How to Use

**Option 1: Simple Version (Recommended)**

Copy the appropriate version below and paste it in Claude Code UI. This lightweight approach leverages the detailed usage strategies built into each MCP tool's description.

**Option 2: Detailed Version**

Copy the content from [CUSTOM_INSTRUCTIONS_CN.md](./CUSTOM_INSTRUCTIONS_CN.md) (Chinese) or [CUSTOM_INSTRUCTIONS_EN.md](./CUSTOM_INSTRUCTIONS_EN.md) (English) and paste it in Claude Code UI. This comprehensive approach includes all processing strategies directly in custom instructions.

---

### Simple Version - Chinese (极简版 - 中文)

```markdown
## 文档处理规则

读取文档文件时,**必须严格遵循以下规则**:

### 必须使用的MCP工具

- PDF文件 -> 使用 `read_pdf` 工具
- Word文档 -> 使用 `read_word` 工具
- Excel表格 -> 使用 `read_excel` 工具
- PowerPoint -> 使用 `read_powerpoint` 工具
- HTML文件 -> 使用 `read_html` 工具
- ZIP压缩包 -> 使用 `read_zip` 工具

### 严格禁止

- **绝对不要**使用Read工具读取上述二进制文件(会得到乱码)
- **绝对不要**尝试其他读取方式

### Read工具的正确用途

Read工具仅适用于纯文本文件(.txt、.md、.py、.sh、.log等)。

MCP工具自动处理LaTeX公式修复、分页、结构化提取等功能。每个工具的描述包含详细使用策略。
```

### Simple Version - English (极简版 - 英文)

```markdown
## Document Processing Rule

When reading document files, **strictly follow these rules**:

### Required MCP Tools

- PDF files -> Use `read_pdf` tool
- Word documents -> Use `read_word` tool
- Excel spreadsheets -> Use `read_excel` tool
- PowerPoint -> Use `read_powerpoint` tool
- HTML files -> Use `read_html` tool
- ZIP archives -> Use `read_zip` tool

### Strictly Prohibited

- **NEVER** use Read tool for above binary files (results in garbled output)
- **NEVER** attempt other reading methods

### Correct Use of Read Tool

Read tool is ONLY for plain text files (.txt, .md, .py, .sh, .log, etc.).

MCP tools automatically handle LaTeX formula fixing, pagination, structured extraction, etc. Each tool's description includes detailed usage strategies.
```

---

### Detailed Version - Chinese (详细版 - 中文)

完整内容请查看: [CUSTOM_INSTRUCTIONS_CN.md](./CUSTOM_INSTRUCTIONS_CN.md)

### Detailed Version - English (详细版 - 英文)

完整内容请查看: [CUSTOM_INSTRUCTIONS_EN.md](./CUSTOM_INSTRUCTIONS_EN.md)
