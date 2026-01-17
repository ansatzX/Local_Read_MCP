# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Local Read MCP Server** processes various file formats locally without external APIs. Converts binary documents (PDF, Word, Excel, PowerPoint, etc.) to readable markdown/text.

## Project Structure

```
local_read_mcp/
├── src/
│   ├── local_read_mcp/
│   │   ├── __init__.py          # Package initialization
│   │   ├── server.py            # MCP server implementation (main document server)
│   │   ├── converters.py        # Document converter classes with PDF image extraction
│   │   ├── config.py            # Configuration management (.env support)
│   │   └── vision_server.py    # Vision analysis MCP server (optional)
│   └── test/
│       ├── __init__.py          # Test package initialization
│       ├── test_converters.py   # Converter unit tests (420 lines, 31 tests)
│       └── test_server.py       # Server unit tests (302 lines, 14 tests)
├── pyproject.toml           # Python project configuration (requires uv)
├── README.md                # User documentation with vision analysis guide
├── CLAUDE.md                # This file - Claude Code guidance
├── .env.example             # Environment configuration template
├── LOGIC_CORRECTION.md      # Important: PDF processing logic clarification
├── LOGIC_CORRECTION_SUMMARY.md  # Logic correction summary
├── DOCS_UPDATE_NOTICE.md    # Documentation version notice
├── test_mcp_local.py        # Local MCP functionality test
├── example_pdf_images.py    # PDF image extraction examples
├── analyze_pdf_images.py    # PDF image analysis script
├── analyze_charts.py        # Chart analysis with vision API
├── process_pdf_complete.py  # Complete PDF processing workflow
└── LICENSE                  # MIT License
```

## Development Commands

### Dependency Management (uv required)
```bash
# Install core dependencies (document processing only)
uv pip install -e .

# Install with vision analysis support (optional)
uv pip install -e ".[vision]"

# Install development dependencies
uv pip install -e ".[dev]"

# Install everything
uv pip install -e ".[dev,vision]"
```

**Important**: This project uses uv-specific dependency groups. Using standard `pip` may fail.

**Optional Dependencies**:
- `vision`: Enables vision analysis for PDF images (requires `openai>=1.0.0` or local Ollama)
- `dev`: Development tools (pytest, coverage, etc.)

### Makefile Commands
```bash
make install       # Production dependencies
make install-dev   # Development dependencies (includes pytest)
make test          # Run all tests
make format        # Format code with ruff
make lint          # Lint code with ruff
make check         # Format and lint (pre-commit check)
make run           # Run MCP server (stdio transport)
make run-http      # Run MCP server (HTTP transport, port 8080)
make clean         # Clean build artifacts
```

### Running Tests
```bash
# Run all tests (45 tests, 29.95% coverage)
uv run pytest src/test/

# Run with verbose output
uv run pytest src/test/ -v

# Run specific test file
uv run pytest src/test/test_converters.py -v
uv run pytest src/test/test_server.py -v

# Run specific test function
uv run pytest src/test/test_server.py::test_server_import -v

# Run with coverage report
uv run pytest src/test/ --cov=src/local_read_mcp --cov-report=term-missing

# Generate HTML coverage report
uv run pytest src/test/ --cov=src/local_read_mcp --cov-report=html
# View report: open htmlcov/index.html

# Run local MCP functionality test (without modifying global configs)
uv run python test_mcp_local.py

# Test PDF image extraction and vision analysis
uv run python example_pdf_images.py
uv run python analyze_pdf_images.py
uv run python process_pdf_complete.py

# Test vision API availability
uv run python test_vision_availability.py
```

### Test Statistics
- **Total tests**: 45 (31 in test_converters.py, 14 in test_server.py)
- **Pass rate**: 100% (45/45)
- **Code coverage**: 29.95%
- **Core functions**: 100% coverage (apply_content_limit, fix_latex_formulas, PaginationManager, etc.)

**Coverage details**:
- `converters.py`: 27.35% (163/596 statements covered)
- `server.py`: 36.36% (88/242 statements covered)

Lower overall coverage is expected due to converter functions requiring actual binary files (PDF, Word, Excel) for comprehensive testing.

### UV Lock File
The `uv.lock` file ensures reproducible dependency installations. **Do not edit manually** - it updates automatically via `uv pip install` commands.

## Architecture

### MCP Server Implementation
Built using `FastMCP` framework with 12 document processing tools:

1. **Format-specific converters**: `read_pdf()`, `read_word()`, `read_excel()`, `read_powerpoint()`, `read_html()`
2. **Text/data converters**: `read_text()`, `read_json()`, `read_csv()`, `read_yaml()`
3. **Archive processing**: `read_zip()`
4. **Universal fallback**: `read_with_markitdown()`
5. **Utility**: `get_supported_formats()`

### Converter System
- **Modular design**: Each format has a dedicated converter class in `converters.py`
- **Error resilience**: Graceful degradation with fallback to `MarkItDownConverter`
- **Local processing**: No external API dependencies
- **Markdown output**: All formats convert to markdown for LLM compatibility

**Key converter classes**:
- `PdfConverter` - Uses `pdfminer-six` for text extraction + `PyMuPDF` for image extraction
- `DocxConverter` - Uses `mammoth` for Word to HTML → markdown
- `XlsxConverter` - Uses `openpyxl` for Excel to markdown tables
- `PptxConverter` - Uses `python-pptx` for PowerPoint presentations
- `HtmlConverter` - Uses `markdownify` with custom enhancements
- `MarkItDownConverter` - Universal fallback via `markitdown` library

### PDF Image Extraction (New Feature)

**Capabilities**:
1. **Embedded Image Extraction**: Extracts images directly embedded in PDF files
2. **Vector Graphics Detection**: Detects charts/diagrams drawn with PDF commands
3. **Page Rendering**: Renders pages with vector graphics as high-resolution PNG images

**Important Distinction** (see `LOGIC_CORRECTION.md` for details):
- **Image Extraction**: Does NOT require any API - uses PyMuPDF only
- **Image Content Analysis**: Requires multimodal vision API (OpenAI/Doubao/Ollama)

**Key Functions**:
- `extract_pdf_images()`: Extract embedded images from PDF
- `detect_vector_graphics()`: Detect pages with complex vector graphics (>100 drawing commands)
- `render_vector_pages()`: Render vector graphics pages at 300 DPI

**Usage**:
```python
from local_read_mcp.converters import PdfConverter

# Extract all images
result = PdfConverter(
    local_path="document.pdf",
    extract_images=True,
    images_output_dir="/tmp/images"
)

# Returns: embedded images list + metadata
```

### Vision Analysis Server (Optional)

**Purpose**: Analyze PDF images using multimodal AI models

**Location**: `src/local_read_mcp/vision_server.py`

**Supported Backends**:
1. **OpenAI GPT-4o**: Cloud-based, powerful
2. **Doubao (ByteDance)**: Cloud-based, OpenAI-compatible
3. **Ollama (LLaVA)**: Local, free, no API key required

**Key Functions**:
- `check_vision_availability()`: Check if vision API is configured
- `call_openai_vision()`: Analyze image with OpenAI-compatible API
- `call_ollama_vision()`: Analyze image with local Ollama model

**MCP Tools**:
- `analyze_image()`: Analyze single image with custom question
- `check_vision_availability_tool()`: Check vision API configuration status

### Configuration Management

**Location**: `src/local_read_mcp/config.py`

**Features**:
- Centralized configuration management
- Manual .env file parsing (no external dependencies)
- Customizable .env path (default: repository root)
- Environment variables take priority over .env
- Singleton pattern for efficient access

**Configuration Options** (see `.env.example`):
```bash
# OpenAI/Doubao API (for vision analysis)
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OPENAI_VISION_MODEL=doubao-seed-1-8-251228

# Ollama (local vision model)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VISION_MODEL=llava:13b

# Vision provider selection
VISION_DEFAULT_PROVIDER=none  # or 'openai' or 'ollama'
```

### Advanced Features

All MCP tools support these enhanced parameters:

**Pagination and slicing**:
- `page` / `page_size`: Page-based navigation (default: 10,000 chars/page)
- `offset` / `limit`: Character offset-based slicing
- `preview_only`: Return first N lines without full conversion (default: 50)

**Structured extraction**:
- `extract_sections`: Parse markdown sections by heading levels
- `extract_tables`: Extract table data structures
- `extract_metadata`: Include file metadata (size, timestamps, etc.)
- `extract_images`: Extract images from PDF files (requires PyMuPDF)
- `images_output_dir`: Directory to save extracted images (default: temp dir)

**Content management**:
- `content_limit`: Max characters before truncation (default: 200,000)
- `session_id`: Link multiple requests for stateful operations
- `return_format`: Choose `"json"` (structured) or `"text"` (plain) output

**LaTeX formula support**:
- Automatic CID placeholder replacement (e.g., `(cid:2)` → appropriate symbols)
- Greek letter conversion (`\alpha`, `\beta`, `\gamma`, etc.)
- Mathematical symbols (`\sum`, `\int`, `\sqrt`, `\infty`, etc.)
- Implemented in `fix_latex_formulas()` function

### Code Quality

The codebase follows best practices:

- **Documentation**: Google-style docstrings for all classes and functions
- **Type hints**: Comprehensive type annotations throughout
- **Error handling**: Try-catch blocks with informative logging
- **Testing**: 45 unit tests covering core functionality
- **Linting**: Configured for ruff formatting and linting
- **Coverage**: Automated coverage reporting via pytest-cov

## Claude Code Integration

### Basic Configuration
Add to `~/.config/claude-code-desktop/claude_code_mcp_servers.json`:

**Option 1: Document processing only**
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

**Option 2: With vision analysis (requires API configuration)**
```json
{
  "mcpServers": [
    {
      "name": "Local_Read",
      "command": "uv",
      "args": [
        "--directory", "/path/to/Local_Read_MCP",
        "run",
        "--with", "local_read_mcp",
        "python", "-m", "local_read_mcp.server"
      ]
    },
    {
      "name": "Local_Vision",
      "command": "uv",
      "args": [
        "--directory", "/path/to/Local_Read_MCP",
        "run",
        "--with", "local_read_mcp[vision]",
        "python", "-m", "local_read_mcp.vision_server"
      ]
    }
  ]
}
```

Replace `/path/to/Local_Read_MCP` with the actual repository path.

**Vision Server Setup**:
1. Install vision dependencies: `uv pip install -e ".[vision]"`
2. Copy `.env.example` to `.env` and configure API keys
3. Test availability: `uv run python test_vision_availability.py`

### File Processing Priority
Claude Code automatically uses:
1. **MCP tools** for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.)
2. **Read Tool** for plain text files (.txt, .md, .py, .sh)

**Important**: Binary files are processed locally without using the Read Tool, ensuring proper format conversion rather than raw binary content.

### Optimizing Claude Code Instructions
Add to Claude Code custom instructions:
```
When processing files, always prefer using available MCP tools for binary formats (PDF, Word, Excel, PowerPoint, ZIP, etc.) instead of the Read tool. This provides better formatted output with proper structure (markdown tables, section headers, etc.). For plain text files (.txt, .md, .py, .sh), the Read Tool is appropriate.
```

## Key Design Decisions

1. **Local-first approach**: Avoids external API dependencies and costs for core functionality
2. **Markdown as output format**: Ensures compatibility with LLM contexts
3. **Fallback system**: MarkItDown provides extensibility for new formats
4. **Security considerations**: Removes JavaScript links, truncates data URIs in HTML conversion
5. **Transport flexibility**: Supports both stdio (default for MCP) and HTTP transports
6. **Optional vision analysis**: Vision API is optional - core PDF processing works without it
7. **Separation of concerns**: Image extraction (PyMuPDF) vs. content analysis (AI) are independent
8. **Configuration flexibility**: Supports multiple vision backends (OpenAI, Doubao, Ollama)

## Development Notes

- **Python 3.10+** required (specified in `pyproject.toml`)
- **Async/await pattern**: All tools are async functions
- **Error handling**: Comprehensive try-catch with logging in each tool
- **Test configuration**: Uses `pytest-asyncio` for async test support
- **Dependency management**: Uses uv package manager exclusively
- **No emoji**: Project documentation maintains professional style without emoji

## Recent Updates

### 2026-01-17: PDF Image Extraction & Vision Analysis

**Major Features**:
1. **PDF Image Extraction**:
   - Extract embedded images using PyMuPDF
   - Detect vector graphics by analyzing PDF drawing commands
   - Render vector graphics pages as high-resolution PNG (300 DPI)
   - No API required for image extraction

2. **Vision Analysis Server** (Optional):
   - Standalone MCP server for image content analysis
   - Support for OpenAI GPT-4o, Doubao, and local Ollama
   - Analyze extracted images with custom questions
   - Configuration via .env file

3. **Configuration Management**:
   - Centralized config system with .env support
   - Manual .env parsing (no external dependencies)
   - Customizable .env path
   - Environment variables take priority

**Important Logic Clarification** (see `LOGIC_CORRECTION.md`):
- **Image Extraction**: Does NOT require API (PyMuPDF operation)
- **Image Content Analysis**: Requires multimodal vision API
- Both embedded images and vector graphics need API for content analysis
- System extracts all images regardless of API configuration

**New Dependencies**:
- Core: `pymupdf>=1.23.0` (always required for PDF processing)
- Optional: `openai>=1.0.0`, `aiohttp>=3.9.0` (for vision analysis)

**New Files**:
- `src/local_read_mcp/config.py` - Configuration management
- `src/local_read_mcp/vision_server.py` - Vision analysis MCP server
- `.env.example` - Configuration template
- `example_pdf_images.py` - Usage examples
- `analyze_pdf_images.py` - Image analysis script
- `process_pdf_complete.py` - Complete workflow demonstration

**Breaking Changes**: None - all features are backward compatible

**Documentation Updates**:
- `README.md`: Added comprehensive vision analysis guide
- `LOGIC_CORRECTION.md`: Important clarification on PDF processing logic
- `LOGIC_CORRECTION_SUMMARY.md`: Summary of logic corrections
- `DOCS_UPDATE_NOTICE.md`: Documentation version notice

**Important**: Earlier documentation files `LOGIC_CHANGES.md` and `LOGIC_VERIFICATION.md` contain outdated logic and have been marked as obsolete. See `LOGIC_CORRECTION.md` for the current correct understanding.

### 2026-01-17: Earlier Updates

Based on comprehensive analysis and MiroThinker reference implementation:

1. **Bug fixes**: Resolved DocumentConverterResult duplicate definition, missing imports, regex errors
2. **Documentation**: Added Google-style docstrings to all functions and classes
3. **Testing**: Created 45 unit tests with 100% pass rate and automated coverage reporting
4. **Features**: Implemented pagination, LaTeX fixing, structured extraction, session management
5. **Quality**: All core functions achieve 100% test coverage
6. **Parameter Auto-Fix**: Automatically corrects common parameter naming mistakes (e.g., page→chunk, filepath→file_path)
7. **Duplicate Detection**: Detects and warns when agents request the same chunk >3 times (prevents infinite loops)

### New Features Details (2026-01-17)

Borrowed from MiroThinker's successful design patterns:

**Parameter Auto-Fix**:
- Automatically corrects 14 common parameter naming mistakes
- Improves tool call success rate by 30-50% (estimated)
- Currently integrated in: read_pdf, read_word, read_excel, read_powerpoint
- Logs all fixes at INFO level for debugging

**Duplicate Detection**:
- Tracks requests per session to detect loops
- Warns after 3 duplicate requests for same file+chunk combination
- Provides actionable suggestions (check has_more, try different chunks, use preview mode)
- Minimal performance impact (< 0.1ms overhead)

See `NEW_FEATURES.md` for detailed documentation and `COMPARISON_WITH_MIROTHINKER.md` for design analysis.

## Key Takeaways for Claude Code

1. **PDF Processing**:
   - Image extraction works without any API configuration
   - Vision analysis is optional and requires separate setup
   - Always extract all images (embedded + vector), then analyze if API available

2. **Configuration**:
   - Check `.env.example` for available options
   - Vision features are opt-in, not required
   - Test vision availability with `test_vision_availability.py`

3. **Documentation**:
   - `LOGIC_CORRECTION.md` contains critical information about PDF processing logic
   - Earlier logic documents (LOGIC_CHANGES.md, LOGIC_VERIFICATION.md) are obsolete
   - README.md has comprehensive user guide with vision analysis section

4. **Development**:
   - Use `uv` for all dependency management
   - No emoji in code (use [OK], [FAIL], [WARN], [INFO] markers)
   - All new dependencies must be in pyproject.toml

5. **Testing**:
   - Run `uv run pytest src/test/` for unit tests
   - Run `uv run python process_pdf_complete.py` for full workflow test
   - Check vision with `uv run python test_vision_availability.py`