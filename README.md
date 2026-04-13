# Local Read MCP Server

Model Context Protocol server for local document processing with vision support.

## 设计理念 / Design Philosophy

### 核心原则

1. **不与 agent 的内置 Read 工具竞争，而是互补**
   - Agent 的内置 Read 用于轻量读取（plain text, images）
   - `process_binary_file` 用于转换二进制文档到结构化格式
   - `query_processed_document` 用于查询已转换的结构化数据

2. **两工具架构**
   ```
   process_binary_file → intermediate.json → query_processed_document
         ↓                    ↓                          ↓
     转换引擎           结构化中间格式          结构化查询引擎
   ```

3. **工作流**
   - 二进制文档：`process_binary_file` → 本地文件夹（intermediate.json + output.md + images/）
   - Agent 用内置 Read 读取 `output.md` 和图片
   - Agent 用 `query_processed_document` 做结构化查询

4. **图片处理**
   - Agent 内置 Read 能直接看图 → 直接用
   - 不能看图 → `process_binary_file` 配 VLM backend → 输出图片描述
   - 图片文件始终本地存储，agent 可调整 prompt 重新分析

## Features

- **Multi-format Support**: PDF, Word, Excel, PowerPoint, HTML, Text, JSON, CSV, YAML, ZIP
- **Local Processing**: No cloud services required
- **Vision Analysis**: OpenAI-compatible API support (Doubao, GPT-4o, etc.)
- **MinerU Integration**: Optional high-quality PDF parsing
- **Markdown Output**: All formats convert to readable markdown

## Quick Start

### Installation

```bash
git clone https://github.com/ansatzX/Local_Read_MCP.git
cd Local_Read_MCP

uv venv .venv
source .venv/bin/activate

uv pip install -e .
uv pip install -e ".[mineru]"  # Optional: MinerU PDF parsing
```

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

## Available Tools

| Tool                    | Description                       |
| ------------------------ | --------------------------------- |
| `read_text_file`         | Read text-based files            |
| `read_binary_file`       | Read binary/document files        |
| `process_text_file`      | Process text files & save to disk|
| `process_binary_file`    | Process binary files & save to disk|
| `analyze_image`          | Analyze images with vision API  |
| `get_vision_status`      | Check vision configuration       |
| `cleanup_temp_files`     | Clean up temporary files         |
| `get_supported_formats`  | List all supported formats        |

## Development

```bash
uv run pytest -q          # Run tests
uv run ruff format .      # Format code
uv run ruff check .       # Lint code
```

## License

MIT License