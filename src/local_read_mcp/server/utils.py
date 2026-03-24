"""Server utility functions for the Local Read MCP Server."""

import logging
import os
import time
from typing import Dict, Optional

from ..converters import (
    DocumentConverterResult,
    apply_content_limit,
    extract_sections_from_markdown
)

logger = logging.getLogger(__name__)


def apply_pagination(content: str, offset: int, limit: Optional[int]) -> tuple[str, bool]:
    """Apply pagination to content."""
    if offset >= len(content):
        return "", False
    if limit:
        end = min(offset + limit, len(content))
        return content[offset:end], end < len(content)
    return content[offset:], False


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

    Examples:
        >>> fix_tool_arguments("read_pdf", {"page": 1, "page_size": 5000})
        {"chunk": 1, "chunk_size": 5000}

        >>> fix_tool_arguments("read_pdf", {"filepath": "/path/to/file.pdf"})
        {"file_path": "/path/to/file.pdf"}
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


class DuplicateDetector:
    """
    Detect duplicate document read requests to prevent infinite loops.

    This class tracks document read requests per session and warns when the same
    file+chunk combination is requested multiple times, which may indicate that
    an agent is stuck in a loop.

    Attributes:
        max_repeats: Maximum number of times a chunk can be requested before warning
        max_sessions: Maximum number of sessions to keep in cache (LRU cleanup)
        request_cache: Dictionary mapping session_id to file+chunk request counts
        session_last_access: Dictionary tracking last access time for each session
    """

    def __init__(self, max_repeats: int = 3, max_sessions: int = 100):
        """
        Initialize the duplicate detector.

        Args:
            max_repeats: Maximum allowed repeats before warning (default: 3)
            max_sessions: Maximum number of sessions to keep in cache (default: 100)
        """
        self.max_repeats = max_repeats
        self.max_sessions = max_sessions
        self.request_cache: Dict[str, Dict[str, int]] = {}
        self.session_last_access: Dict[str, float] = {}

    def _cleanup_old_sessions(self):
        """Clean up oldest sessions if we exceed max_sessions limit."""
        if len(self.request_cache) <= self.max_sessions:
            return

        # Sort sessions by last access time (oldest first)
        sorted_sessions = sorted(
            self.session_last_access.items(),
            key=lambda x: x[1]
        )

        # Remove oldest sessions
        sessions_to_remove = len(self.request_cache) - self.max_sessions
        for session_id, _ in sorted_sessions[:sessions_to_remove]:
            if session_id in self.request_cache:
                del self.request_cache[session_id]
            if session_id in self.session_last_access:
                del self.session_last_access[session_id]
            logger.debug(f"[Duplicate Detection] Cleaned up old session: {session_id[:8]}...")

    def check_and_record(
        self,
        session_id: str,
        file_path: str,
        chunk: int,
        chunk_size: int
    ) -> Optional[str]:
        """
        Check if this is a duplicate request and record it.

        Args:
            session_id: Session identifier
            file_path: Path to the file being read
            chunk: Chunk number being requested
            chunk_size: Size of the chunk

        Returns:
            Warning message if duplicate detected, None otherwise

        Examples:
            >>> detector = DuplicateDetector(max_repeats=3)
            >>> detector.check_and_record("sess1", "/file.pdf", 1, 10000)
            None  # First request, no warning

            >>> # After 3 more requests for the same chunk...
            >>> detector.check_and_record("sess1", "/file.pdf", 1, 10000)
            "Warning: This chunk has been requested 3 times..."
        """
        # Clean up old sessions if needed
        self._cleanup_old_sessions()

        # Update session last access time
        self.session_last_access[session_id] = time.time()

        # Create cache key: file_path + chunk + chunk_size
        cache_key = f"{file_path}:chunk{chunk}:size{chunk_size}"

        # Initialize session cache if needed
        if session_id not in self.request_cache:
            self.request_cache[session_id] = {}

        # Get current count
        count = self.request_cache[session_id].get(cache_key, 0)

        # Check if exceeded max repeats
        if count >= self.max_repeats:
            warning_msg = (
                f"Warning: Chunk {chunk} of '{os.path.basename(file_path)}' has been "
                f"requested {count} times in this session. You may be in a loop. "
                f"Suggestions: (1) Check if has_more=False, (2) Try different chunk numbers, "
                f"(3) Use preview_only=True to assess content first."
            )
            logger.warning(f"[Duplicate Detection] {warning_msg}")
            # Still record the request
            self.request_cache[session_id][cache_key] = count + 1
            return warning_msg

        # Record the request
        self.request_cache[session_id][cache_key] = count + 1

        # Log for debugging (only for repeat requests)
        if count > 0:
            logger.info(
                f"[Duplicate Detection] Chunk {chunk} of '{os.path.basename(file_path)}' "
                f"requested {count + 1} times in session {session_id[:8]}..."
            )

        return None

    def clear_session(self, session_id: str):
        """
        Clear cache for a specific session.

        Args:
            session_id: Session identifier to clear
        """
        if session_id in self.request_cache:
            del self.request_cache[session_id]
        if session_id in self.session_last_access:
            del self.session_last_access[session_id]
        logger.info(f"[Duplicate Detection] Cleared session {session_id[:8]}...")

    def get_session_stats(self, session_id: str) -> Dict[str, int]:
        """
        Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary mapping cache keys to request counts
        """
        return self.request_cache.get(session_id, {}).copy()


# Global duplicate detector instance
duplicate_detector = DuplicateDetector(max_repeats=3, max_sessions=100)


def create_simple_converter_wrapper(converter_func, converter_name: str = ""):
    """Create a wrapper for simple converters that don't support enhanced parameters.

    Args:
        converter_func: The original converter function
        converter_name: Name of the converter for logging

    Returns:
        A wrapper function that supports extract_metadata, extract_sections, extract_tables
    """
    def wrapper(file_path: str, **kwargs) -> DocumentConverterResult:
        # Call the original converter function
        result = converter_func(file_path)

        # Apply content limit (200,000 characters) like other converters
        result.text_content = apply_content_limit(result.text_content)

        # If extract_metadata is requested, add file metadata
        if kwargs.get('extract_metadata', False):
            file_size = None
            try:
                file_size = os.path.getsize(file_path)
            except (OSError, Exception):
                pass
            result.metadata = {
                "file_path": file_path,
                "file_size": file_size,
                "file_extension": os.path.splitext(file_path)[1],
            }

        # If extract_sections is requested, extract sections from markdown
        if kwargs.get('extract_sections', False):
            result.sections = extract_sections_from_markdown(result.text_content)

        # If extract_tables is requested
        if kwargs.get('extract_tables', False):
            result.tables = []  # No table extraction for simple converters

        return result

    return wrapper
