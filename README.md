# Local Read MCP Server

Model Context Protocol server for local document processing with vision support.

## Features

- **Consolidated Tools**: Two main tools: `read_text_file` and `read_binary_file`
- **Local Processing**: No cloud services required for document conversion
- **Multi-format Support**: PDF, Word, Excel, PowerPoint, HTML, Text, JSON, CSV, YAML, ZIP
- **Enhanced PDF Features**: Rendering, table extraction, form handling, metadata inspection, coordinate-aware text extraction
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

# Install basic dependencies
uv pip install -e .

# OR install with enhanced PDF features (includes pdfplumber for table extraction)
uv pip install -e ".[pdf]"
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
- `OPENAI_VISION_MODEL` instead of `VISION_MODEL`

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

| Tool                    | Description                       | Status       |
| ------------------------ | --------------------------------- | ------------ |
| `read_text_file`         | Read text-based files            | **Main**     |
| `read_binary_file`       | Read binary/document files        | **Main**     |
| `analyze_image`          | Analyze images with vision API  | Auxiliary    |
| `get_vision_status`      | Check vision configuration       | Auxiliary    |
| `cleanup_temp_files`     | Clean up temporary files         | Auxiliary    |
| `get_supported_formats`  | List all supported formats        | Info         |

### Deprecated Tools (for backward compatibility)

| Tool                    | Migration Guide                   |
| ------------------------ | --------------------------------- |
| `read_pdf`             | Use `read_binary_file` or `read_binary_file(format='pdf')` |
| `read_word`            | Use `read_binary_file` or `read_binary_file(format='word')` |
| `read_excel`           | Use `read_binary_file` or `read_binary_file(format='excel')` |
| `read_powerpoint`       | Use `read_binary_file` or `read_binary_file(format='ppt')` |
| `read_html`             | Use `read_binary_file` or `read_binary_file(format='html')` |
| `read_text`             | Use `read_text_file` or `read_text_file(format='text')` |
| `read_json`             | Use `read_text_file` or `read_text_file(format='json')` |
| `read_csv`              | Use `read_text_file` or `read_text_file(format='csv')` |
| `read_yaml`             | Use `read_text_file` or `read_text_file(format='yaml')` |
| `read_zip`              | Use `read_binary_file` or `read_binary_file(format='zip')` |

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

### PDF Enhanced Parameters (read_binary_file with format=pdf only)

| Parameter          | Type  | Default | Description                          |
| ----------------- | ----- | ------- | ------------------------------------ |
| `render_images`    | bool  | False    | Render PDF pages to images           |
| `render_dpi`       | int   | 200      | DPI for rendered images              |
| `render_format`    | str   | "png"    | Image format (png/jpeg)              |
| `extract_forms`    | bool  | False    | Extract form fields with types/values/positions |
| `inspect_struct`   | bool  | False    | Get complete PDF structure/metadata/outline/fonts |
| `include_coords`   | bool  | False    | Include bounding box coordinates with text |

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

### Text File Reading

```python
# Simple text file - auto-detects format
read_text_file("/path/to/document.txt")
read_text_file("/path/to/data.json")
read_text_file("/path/to/table.csv")

# Explicit format override
read_text_file("/path/to/file", format="json")
read_text_file("/path/to/file", format="csv")

# With pagination
read_text_file("/path/to/large.md", chunk=1, chunk_size=10000)

# Extract tables (CSV)
read_text_file("/path/to/table.csv", extract_tables=True, return_format="json")
```

### Binary/Document File Reading

```python
# Simple - auto-detects format
read_binary_file("/path/to/document.pdf")
read_binary_file("/path/to/spreadsheet.xlsx")
read_binary_file("/path/to/archive.zip")

# Explicit format override
read_binary_file("/path/to/file", format="pdf")
read_binary_file("/path/to/file", format="excel")

# With pagination
read_binary_file("/path/to/large.pdf", chunk=1, chunk_size=10000)

# PDF with enhanced features
read_binary_file(
    "/path/to/document.pdf",
    extract_images=True,
    render_images=True,
    extract_tables=True,
    extract_forms=True,
    inspect_struct=True,
    return_format="json"
)
```

### Vision Analysis

```python
# Check if vision is configured
status = get_vision_status()

# Analyze image
if status["vision_enabled"]:
    result = analyze_image("/path/to/chart.png", "Describe this chart")
```

### Temporary Files Cleanup

```python
# Dry run - see what would be deleted without actually deleting
cleanup_temp_files(dry_run=True)

# Clean up temporary files older than 24 hours (default)
cleanup_temp_files()

# Clean up all temporary files immediately
cleanup_temp_files(older_than_hours=0)

# Clean up only PDF image extraction directories
cleanup_temp_files(cleanup_zip_extracts=False)

# Clean up a custom directory in addition to temp dirs
cleanup_temp_files(custom_directory="/path/to/your/dir")
```

**Note:** Temporary files are created when using:
- `read_binary_file(extract_images=True)` - PDF image extraction
- `read_binary_file(render_images=True)` - PDF page rendering
- `read_binary_file(format="zip")` - ZIP extraction

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

- 文本文件(.txt, .md, .py, .sh, .json, .csv, .yaml, .yml) -> 使用 `read_text_file` 工具
- PDF文件 -> 使用 `read_binary_file` 工具 (或 `read_binary_file(format='pdf')`)
- Word文档 -> 使用 `read_binary_file` 工具 (或 `read_binary_file(format='word')`)
- Excel表格 -> 使用 `read_binary_file` 工具 (或 `read_binary_file(format='excel')`)
- PowerPoint -> 使用 `read_binary_file` 工具 (或 `read_binary_file(format='ppt')`)
- HTML文件 -> 使用 `read_binary_file` 工具 (或 `read_binary_file(format='html')`)
- ZIP压缩包 -> 使用 `read_binary_file` 工具 (或 `read_binary_file(format='zip')`)

### 临时文件清理

- **重要**: 使用 `read_binary_file(extract_images=True)`、`read_binary_file(render_images=True)` 或 `read_binary_file(format='zip')` 后会产生临时文件
- **任务结束后**: 确认用户不再需要这些文件时，使用 `cleanup_temp_files` 工具清理
- 默认清理24小时前的文件，可使用 `older_than_hours=0` 立即清理所有

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

- Text files (.txt, .md, .py, .sh, .json, .csv, .yaml, .yml) -> Use `read_text_file` tool
- PDF files -> Use `read_binary_file` tool (or `read_binary_file(format='pdf')`)
- Word documents -> Use `read_binary_file` tool (or `read_binary_file(format='word')`)
- Excel spreadsheets -> Use `read_binary_file` tool (or `read_binary_file(format='excel')`)
- PowerPoint -> Use `read_binary_file` tool (or `read_binary_file(format='ppt')`)
- HTML files -> Use `read_binary_file` tool (or `read_binary_file(format='html')`)
- ZIP archives -> Use `read_binary_file` tool (or `read_binary_file(format='zip')`)

### Temporary Files Cleanup

- **IMPORTANT**: Using `read_binary_file(extract_images=True)`, `read_binary_file(render_images=True)`, or `read_binary_file(format='zip')` creates temporary files
- **After task completion**: When user confirms no further need for these files, use `cleanup_temp_files` tool to clean up
- Default cleans files older than 24 hours, use `older_than_hours=0` to clean all immediately

### Strictly Prohibited

- **NEVER** use Read tool for above binary files (results in garbled output)
- **NEVER** attempt other reading methods

### Correct Use of Read Tool

Read tool is ONLY for plain text files (.txt, .md, .py, .sh, .log, etc.).

MCP tools automatically handle LaTeX formula fixing, pagination, structured extraction, etc. Each tool's description includes detailed usage strategies.
```
