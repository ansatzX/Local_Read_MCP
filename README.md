# Local Read MCP Server

MCP server for local document processing — structured extraction from PDFs, Office, images, and code. Designed to complement an agent's built-in Read tool, not duplicate it.

## Tools

| Tool | When to use |
|------|-------------|
| `process_binary_file` | **MUST** for any non-text file before reading. Converts to structured output and saves to `.local_read_mcp/`. |
| `analyze_image` | Analyze images via Vision API (Doubao, GPT-4o, etc.). Result saved to `.local_read_mcp/analysis/`. |
| `get_vision_status` | Check if Vision API is configured. |

## Quick Start

```bash
git clone https://github.com/ansatzX/Local_Read_MCP.git
cd Local_Read_MCP

uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

Configure MCP in `~/.claude/settings.json`:

```json
{
  "mcpServers": [{
    "command": "uv",
    "args": ["--directory", "/path/to/Local_Read_MCP", "run", "python", "-m", "local_read_mcp.server"]
  }]
}
```

## Vision API (Optional)

Create `.env`:

```
VISION_API_KEY=sk-xxx
VISION_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VISION_MODEL=doubao-seed-1-8-251228
```

## MinerU VLM-HYBRID Backend (Optional)

For high-quality PDF parsing with layout analysis, formula recognition, and table detection:

```bash
pip install "local-read-mcp[mineru]"

# Download models into the project directory
mineru-models-download --models-dir ./models

# Configure model paths
cp mineru.json.template mineru.json
# Edit mineru.json if models are elsewhere
```

Models downloaded (~4.5GB total):
- `models/pipeline/` — layout detection, OCR, formula recognition, table structure
- `models/vlm/` — fine-tuned Qwen2-VL for document understanding

## How It Works

```
process_binary_file(file.pdf)
  ├─ format detection → backend selection
  │
  ├─ SIMPLE (zero-dependency, all formats)
  │   └─ Built-in converters: PyMuPDF / mammoth / openpyxl / python-pptx
  │
  ├─ VLM-HYBRID (MinerU required, PDF only)
  │   └─ MinerU hybrid-auto-engine: VLM layout + pipeline OCR/formula/table
  │
  └─ chapter_split (auto for large PDFs)
      ├─ Detect sections via TOC / heading scan / fixed chunks
      ├─ Process each chunk independently (with page overlap)
      └─ Merge output.md + structural_toc.json
```

All results saved to `.local_read_mcp/<file>_<timestamp>/`:
- `intermediate.json` — structured block representation
- `output.md` — converted markdown
- `index.json` — section/table/figure index
- `images/` — extracted images (when requested)

## Development

```bash
uv run pytest          # 136 tests
uv run ruff format .   # Format code
uv run ruff check .    # Lint
```

## License

MIT
