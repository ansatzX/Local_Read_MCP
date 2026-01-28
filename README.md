# Local Read MCP Server

Model Context Protocol server for local document processing with vision support.

## Features

- **Local Processing**: No cloud services required for document conversion
- **Multi-format Support**: PDF, Word, Excel, PowerPoint, HTML, Text, JSON, CSV, YAML, ZIP
- **Markdown Output**: All formats convert to readable markdown
- **MCP Integration**: Seamless with Claude Code and other MCP clients
- **Vision Analysis**: Analyze images using OpenAI-compatible APIs (Doubao, GPT-4o, etc.)

## Quick Start

```bash
git clone https://github.com/ansatzX/Local_Read_MCP.git
cd Local_Read_MCP

# Install uv if not already installed: https://github.com/astral-sh/uv
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (includes vision support)
uv pip install -e .
```

## Configuration

### Vision API (Optional)

Create `.env` file for vision analysis:

```bash
# Copy example
cp .env.example .env

# Edit .env - use your preferred API
VISION_API_KEY=your-api-key
VISION_BASE_URL=https://api.openai.com/v1
VISION_MODEL=gpt-4o
```

**Supported APIs**:
- **Doubao**: `VISION_BASE_URL=https://ark.cn-beijing.volces.com/api/v3`
- **OpenAI**: `VISION_BASE_URL=https://api.openai.com/v1`
- **Any OpenAI-compatible API**: Set appropriate URL and model

**Alternative variable names** (for compatibility):
- `OPENAI_API_KEY` instead of `VISION_API_KEY`
- `OPENAI_BASE_URL` instead of `VISION_BASE_URL`
- `OPENAI_VISIONAPI_MODEL` instead of `VISION_MODEL`

### Claude Code Setup

Edit `~/.claude/settings.json`:

```json
{
  "mcpServers": [
    {
      "command": "uv",
      "args": [
        "--directory", "/path/to/Local_Read_MCP",
        "run", "--with", "local_read_mcp",
        "python", "-m", "local_read_mcp.server"
      ]
    }
  ]
}
```

Restart Claude Code to load the server.

## Available Tools

| Tool                    | Description                       | Formats                     |
| ------------------------ | --------------------------------- | ---------------------------- |
| `read_pdf`             | PDF to markdown                  | .pdf                        |
| `read_word`            | Word to markdown                 | .docx, .doc                |
| `read_excel`           | Excel to markdown tables           | .xlsx, .xls                |
| `read_powerpoint`       | PowerPoint to markdown            | .pptx, .ppt                |
| `read_html`             | HTML to markdown                 | .html, .htm                 |
| `read_text`             | Plain text files                 | .txt, .md, .py, .sh        |
| `read_json`             | Parse JSON                      | .json                        |
| `read_csv`              | CSV to markdown tables            | .csv                         |
| `read_yaml`             | Parse YAML                       | .yaml, .yml                 |
| `read_zip`              | List ZIP contents                | .zip                         |
| `read_with_markitdown`   | Universal fallback                | Many formats                 |
| `get_supported_formats`  | List all supported formats        | -                            |
| `analyze_image`          | Analyze images with vision API  | .jpg, .png, .gif, etc.    |
| `get_vision_status`      | Check vision configuration       | -                            |

### Common Parameters

| Parameter          | Type  | Default | Description                          |
| ----------------- | ----- | ------- | ------------------------------------ |
| `chunk`            | int   | 1        | Page number for pagination (1-indexed) |
| `chunk_size`       | int   | 10000    | Characters per page                    |
| `offset`           | int   | None     | Character offset (alternative to chunk)   |
| `limit`            | int   | None     | Character limit (alternative to chunk_size) |
| `extract_sections` | bool  | False    | Extract document sections/headings        |
| `extract_tables`  | bool  | False    | Extract table information                 |
| `extract_metadata` | bool  | False    | Extract file metadata                   |
| `extract_images`  | bool  | False    | Extract images (PDF only)              |
| `preview_only`     | bool  | False    | Return preview (first N lines)          |
| `return_format`    | str   | "text"   | Output: "text" or "json"            |

### Vision Tools

**analyze_image(image_path, question, api_key)**

Analyze an image using OpenAI-compatible vision API.

```python
# Basic usage
analyze_image("/path/to/image.png")

# With custom question
analyze_image("/path/to/chart.png", "What data does this chart show?")
```

**get_vision_status()**

Check vision configuration status.

```python
{
  "vision_enabled": true,
  "message": "Vision features available",
  "configured": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "has_api_key": true,
    "max_image_size_mb": 20
  }
}
```

## Usage Examples

### PDF Reading

```python
# Simple
read_pdf("/path/to/document.pdf")

# With pagination
read_pdf("/path/to/large.pdf", chunk=1, chunk_size=10000)

# Extract images
read_pdf("/path/to/document.pdf", extract_images=True)
```

### Vision Analysis

```python
# Check if vision is configured
status = get_vision_status()

# Analyze image
if status["vision_enabled"]:
    result = analyze_image("/path/to/chart.png", "Describe this chart")
```

## Development

```bash
# Run tests
uv run pytest src/test/ -v

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

## License

MIT License

---

## Claude Code Custom Instructions

### Simple Version (Chinese)

```markdown
## 文档处理规则

读取文档文件时，**必须严格遵循以下规则**：

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

### Simple Version (English)

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
