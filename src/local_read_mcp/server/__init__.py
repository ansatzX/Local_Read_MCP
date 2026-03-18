# Local Read MCP Server
# A Model Context Protocol server for document processing

from .app import mcp, main, process_document
from .vision import guess_mime_type_from_extension, call_vision_api
from .utils import (
    apply_pagination,
    fix_tool_arguments,
    DuplicateDetector,
    duplicate_detector,
    create_simple_converter_wrapper
)

__all__ = [
    # Main server
    "mcp",
    "main",
    "process_document",

    # Vision
    "guess_mime_type_from_extension",
    "call_vision_api",

    # Utils
    "apply_pagination",
    "fix_tool_arguments",
    "DuplicateDetector",
    "duplicate_detector",
    "create_simple_converter_wrapper",
]
