# Local_Read_MCP Examples

This directory contains examples of using Local_Read_MCP.

---

## Basic Usage

### Using the CLI

```bash
# Convert a document (see README for actual CLI usage)
local-read-mcp document.pdf
```

### Using as a Library

```python
from pathlib import Path
from local_read_mcp.backends import get_registry, BackendType
from local_read_mcp.output_manager import OutputManager
from local_read_mcp.markdown_converter import MarkdownConverter

# Get registry and select backend
registry = get_registry()
backend = registry.select_best()

# Process a file
file_path = Path("document.pdf")
result = backend.process(file_path, "pdf")

# Create output directory
output_manager = OutputManager()
output_path = output_manager.create_output_dir(str(file_path))

# Save intermediate JSON
import json
intermediate_path = output_manager.get_output_path(output_path, "intermediate.json")
with open(intermediate_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# Convert to Markdown
md_converter = MarkdownConverter(result)
markdown_content = md_converter.convert()

# Save Markdown
markdown_path = output_manager.get_output_path(output_path, "output.md")
with open(markdown_path, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"Files saved to: {output_path}")
```

---

## Backend-specific Examples

### Using Simple Backend

```python
from local_read_mcp.backends import get_registry, BackendType

registry = get_registry()
backend = registry.get(BackendType.SIMPLE)

# Process any supported format
result = backend.process(Path("document.pdf"), "pdf")
```

### Using MinerU Backend

```python
from local_read_mcp.backends import get_registry, BackendType

registry = get_registry()
backend = registry.get(BackendType.MINERU)

if backend and backend.available:
    # Process with formula and table recognition
    result = backend.process(
        Path("paper.pdf"),
        "pdf",
        formula_enable=True,
        table_enable=True,
        language="en"
    )
elif backend:
    print(f"MinerU not available: {backend.warning}")
```

### Using OpenAI VLM Backend

```python
from local_read_mcp.backends import get_registry, BackendType

registry = get_registry()
backend = registry.get(BackendType.OPENAI_VLM)

if backend and backend.available:
    # Process a scanned document
    result = backend.process(
        Path("scanned.pdf"),
        "pdf",
        max_pages=10,
        prompt="Extract all text from this document"
    )
elif backend:
    print(f"OpenAI VLM not available: {backend.warning}")
```

### Using Qwen-VL Backend

```python
from local_read_mcp.backends import get_registry, BackendType

registry = get_registry()
backend = registry.get(BackendType.QWEN_VL)

if backend and backend.available:
    # Process a Chinese document
    result = backend.process(
        Path("chinese_document.pdf"),
        "pdf",
        max_pages=5
    )
elif backend:
    print(f"Qwen-VL not available: {backend.warning}")
```

---

## Caching Example

```python
from pathlib import Path
from local_read_mcp.cache import get_cache_manager
from local_read_mcp.backends import get_registry

# Get cache manager
cache = get_cache_manager(default_ttl=7200)  # 2 hours

# Get backend
registry = get_registry()
backend = registry.select_best()

file_path = Path("document.pdf")

# Check cache first
cached = cache.get(file_path, backend=backend.name)
if cached:
    print("Using cached result!")
    result = cached
else:
    # Process file
    print("Processing file...")
    result = backend.process(file_path, "pdf")
    # Cache the result
    cache.set(file_path, result, backend=backend.name)

# Use the result...
```

---

## PDF Classification Example

```python
from pathlib import Path
from local_read_mcp.mineru import classify_pdf

# Classify a PDF
pdf_path = Path("document.pdf")
classification = classify_pdf(pdf_path)

print(f"PDF classification: {classification}")

if classification == "txt":
    print("Text-based PDF - use text extraction")
else:
    print("Scanned PDF - use OCR")
```

---

## See Also

- [Backend Documentation](../docs/backends.md) - Complete backend documentation
- [README](../README.md) - Main project README