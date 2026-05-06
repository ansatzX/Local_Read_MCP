"""Server utility functions for the Local Read MCP Server."""

import logging

logger = logging.getLogger(__name__)


def fix_tool_arguments(tool_name: str, arguments: dict) -> dict:
    """
    Auto-fix common parameter name mistakes made by LLM.

    This function automatically corrects common parameter naming errors that LLMs
    make when calling tools, improving the success rate of tool calls.

    Args:
        tool_name: Name of the tool being called
        arguments: Original arguments dictionary

    Returns:
        Fixed arguments dictionary with corrected parameter names
    """
    fixed = arguments.copy()

    # Historical parameter name compatibility (page -> chunk migration)
    renames = {
        "page": "chunk",
        "page_size": "chunk_size",
        "pagesize": "chunk_size",
        "pages": "chunk",
        # Common file path variations
        "filepath": "file_path",
        "path": "file_path",
        "file": "file_path",
        "filename": "file_path",
        # Format variations
        "format": "return_format",
        "output_format": "return_format",
        # Other common mistakes
        "preview": "preview_only",
        "metadata": "extract_metadata",
        "sections": "extract_sections",
        "tables": "extract_tables",
        "images": "extract_images",
        "image_dir": "images_output_dir",
        "output_dir": "images_output_dir",
    }

    for old_name, new_name in renames.items():
        if old_name in fixed and new_name not in fixed:
            fixed[new_name] = fixed.pop(old_name)
            logger.info(f"[Parameter Auto-Fix] {tool_name}: '{old_name}' → '{new_name}'")

    return fixed
