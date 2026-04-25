# Local Read MCP — Architecture

## Tools

Three MCP tools:

| Tool | Purpose |
|------|---------|
| `process_binary_file` | **MUST** for any non-text file. Converts to structured output and saves to `.local_read_mcp/`. |
| `analyze_image` | Vision API analysis of images, result saved to `.local_read_mcp/analysis/`. |
| `get_vision_status` | Check if Vision API is configured. |

All output is written to `.local_read_mcp/` in the current working directory. No files are written outside the working directory.

## Architecture

```
process_binary_file(file)
  │
  ├─ format detection → BackendRegistry.select_best()
  │    priority: VLM_HYBRID > SIMPLE
  │
  ├─ SIMPLE  (zero-dependency, all formats)
  │   └─ Built-in converters: PyMuPDF, mammoth, openpyxl, python-pptx, etc.
  │
  ├─ VLM_HYBRID  (requires MinerU + models, PDF only)
  │   └─ MinerU hybrid-auto-engine
  │       VLM layout → pipeline OCR/formula/table → middle_json
  │       Engine: vLLM > LMDeploy > MLX-VLM > transformers (auto)
  │
  └─ chapter_split (internal, triggers for large PDFs)
      ├─ TocExtractor: PyMuPDF TOC → page label calibration → fixed chunk fallback
      ├─ ChunkPlanner: page ranges with overlap
      ├─ per-chunk backend processing → sliced PDF → intermediate.json
      └─ merged output.md + structural_toc.json

Result → .local_read_mcp/<file>_<timestamp>/
  ├── intermediate.json    (structured block representation)
  ├── output.md            (markdown conversion)
  ├── index.json           (section/table/figure index)
  └── images/              (extracted images)
```

## Backend System

```python
class BackendType(Enum):
    AUTO = "auto"
    SIMPLE = "simple"
    VLM_HYBRID = "vlm-hybrid"
```

- **SIMPLE**: Always available, handles all formats. Uses built-in converters.
- **VLM_HYBRID**: PDF only, requires MinerU + downloaded models. Calls `hybrid_analyze.doc_analyze()` directly — no callbacks, no tempdir I/O.
- **Selection**: `VLM_HYBRID > SIMPLE` (by available + format support).

## Chapter Detection (`src/local_read_mcp/segmenter/`)

Built into `process_binary_file`. Triggers when `chapter_split != False` and format is PDF.

Calibration of logical page numbers (TOC) to physical page indices:
1. `page.get_label()` — PDF /PageLabels structure
2. Heuristic text scan — match chapter titles in candidate pages
3. `logical - 1` — fallback

Chunk planning with configurable `overlap` and `min_chunk_pages`. Falls back to fixed-size chunks when no TOC or headings are detected.

## MinerU Integration

MinerU is an external dependency (`pip install local-read-mcp[mineru]`). Models (~4.5GB total) are downloaded by MinerU's own tool, configured via `mineru.json` in the project root. The backend sets `MINERU_TOOLS_CONFIG_JSON` automatically at import time.

Calls MinerU APIs directly:
- `hybrid_analyze.doc_analyze()` for VLM-HYBRID
- Engine auto-selection by MinerU's `get_vlm_engine()`
- Output converted from MinerU's `middle_json` to `IntermediateJSON`

## Configuration Files

| File | Purpose | Tracked |
|------|---------|:-------:|
| `.env` | Vision API key, base URL, model | gitignored |
| `.env.example` | Template for .env | yes |
| `mineru.json` | MinerU model paths | gitignored |
| `mineru.json.template` | Template for mineru.json | yes |

## Source Layout

```
src/local_read_mcp/
├── server/app.py           MCP tools + orchestration
├── backends/
│   ├── base.py             BackendType enum, registry
│   ├── simple.py           SimpleBackend (all formats)
│   └── mineru.py           VlmHybridBackend (MinerU hybrid)
├── segmenter/
│   ├── toc_extractor.py    TOC extraction + page calibration
│   └── chunk_planner.py    Page range planning + overlap
├── converters/             Format converters (PyMuPDF, mammoth, etc.)
├── output_manager.py       .local_read_mcp/ directory management
├── markdown_converter.py   Intermediate JSON → markdown
├── index_generator.py      Section/table/figure index
└── intermediate_json.py    Structured intermediate representation
```
