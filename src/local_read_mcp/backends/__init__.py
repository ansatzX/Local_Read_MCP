# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
Backend implementations for Local_Read_MCP.
"""

from .base import (
    BackendType,
    DocumentBackend,
    BackendRegistry,
    get_registry,
    register_simple_backend,
    register_mineru_backend,
    register_vlm_backends
)

__all__ = [
    "BackendType",
    "DocumentBackend",
    "BackendRegistry",
    "get_registry",
    "register_simple_backend",
    "register_mineru_backend",
    "register_vlm_backends"
]

# Register SimpleBackend
register_simple_backend()

# Register MinerUBackend
register_mineru_backend()

# Register VLM backends
register_vlm_backends()
