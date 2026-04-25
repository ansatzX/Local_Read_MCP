# AGENTS.md — for AI agents working on this codebase

## Project Identity

Local Read MCP is a document processing MCP server. Its purpose: let AI agents read local files that a built-in Read tool cannot handle (PDFs, Office docs, images, ZIPs) and get structured output saved to disk.

## Design Principles

### 1. Complement, don't duplicate

Agents already have a built-in Read tool for plain text and images. This project does **not** add redundant read tools. It adds what's missing:
- `process_binary_file` — **MUST** for any binary file before reading. This is the only entry point.
- `analyze_image` — vision API analysis of images, result saved to disk.
- `get_vision_status` — check if vision API is configured.

### 2. Output always goes to .local_read_mcp/

All persistent output from any tool must land inside `.local_read_mcp/` in the current working directory. Never write to temporary directories, system paths, or anywhere outside the working directory. No cleanup tools needed — nothing to clean up.

### 3. Backend selection is layered

```
BackendType: AUTO → SIMPLE → VLM_HYBRID
```

- `SIMPLE` — zero external dependencies, handles all formats. Always available.
- `VLM_HYBRID` — requires MinerU `pip install local-read-mcp[mineru]` + downloaded models (~4.5GB). PDF only. Calls MinerU's `hybrid_analyze.doc_analyze()` directly — never through `do_parse`'s callback/tempdir pattern.

Select best is `VLM_HYBRID > SIMPLE`. No other backends exist. `OPENAI_VLM` and `QWEN_VL` were removed — they were duplicates. General-purpose vision API (GPT-4V, Doubao) belongs in `analyze_image`, not in the backend system.

### 4. Chapter splitting is built in

`process_binary_file` has `chapter_split`, `start_page`, `end_page`, `page_batch_size` parameters. For large PDFs (>30 pages default), the segmenter module automatically:
1. Extracts TOC via PyMuPDF (or scans text, or falls back to fixed chunks)
2. Calibrates logical→physical page numbers (3-tier: /PageLabels → heuristic → logical-1)
3. Plans chunks with configurable overlap
4. Processes each chunk independently through the backend
5. Merges output.md + structural_toc.json

This is internal, not an extra tool. Single `process_binary_file` call handles everything.

### 5. MinerU is a dependency, not forked code

MinerU is installed as `pip install mineru`. We call its Python API directly:
- `hybrid_analyze.doc_analyze()` for VLM-HYBRID backend
- `vlm_analyze.doc_analyze()` if needed
- `pipeline_analyze.doc_analyze_streaming()` if needed

We do NOT fork MinerU code, reimplement its models, or manage model downloads. Model paths are configured in `mineru.json` (project root, gitignored). Engine selection (vLLM > LMDeploy > MLX-VLM > transformers) is handled by MinerU's `get_vlm_engine()`.

MinerU's config is pointed to the project root automatically via `os.environ.setdefault("MINERU_TOOLS_CONFIG_JSON", ...)` in `backends/mineru.py`.

## Code Conventions

- **Imports**: Use relative imports within the package (`from ..backends import ...`). Only use absolute imports in tests.
- **Docstrings**: Keep them short for MCP tools — the docstring is sent as tool description context to the LLM. No Args sections longer than needed.
- **Error handling**: Backend processing failures should warn and fall back, not crash. Outer `process_binary_file` catches exceptions and returns `{"success": False, "error": ...}`.
- **Tests**: 136 tests, `pytest` with `asyncio` support. Test with real file flows (Simple backend), mock MinerU (it's optional).
- **Architecture docs**: After any change to the tool set, backend system, output layout, or configuration, update `ARCHITECTURE.md` to describe the new state. Keep it current; no historical "what changed" sections.

## Key Architectural Decisions (Why)

| Decision | Why |
|----------|-----|
| Only 2 backends instead of 4 | OPENAI_VLM and QWEN_VL were identical code, just renamed |
| MinerU backend calls hybrid analyzer directly | do_parse uses callback pattern (pipeline only) + writes to disk; we want direct return + control output |
| Output always .local_read_mcp/ | Prevents scattered temp files; no cleanup tool needed |
| chapter_split built into process_binary_file | Single call for all cases; no extra tool for the LLM to figure out |
| mineru.json in project root | No dependency on home directory config; portable setup |
| VLM-HYBRID as single backend name | MinerU's three internal paths (pipeline/vlm/hybrid) are MinerU's concern, not ours |

## Files to Know

| File | Purpose |
|------|---------|
| `src/local_read_mcp/server/app.py` | MCP tools, orchestration |
| `src/local_read_mcp/backends/mineru.py` | VlmHybridBackend → MinerU hybrid analyzer |
| `src/local_read_mcp/backends/base.py` | BackendType enum, registry |
| `src/local_read_mcp/segmenter/` | TocExtractor, ChunkPlanner |
| `src/local_read_mcp/converters/` | All format converters (Simple backend) |
| `mineru.json.template` | MinerU config template |
| `ARCHITECTURE.md` | Full design doc |
