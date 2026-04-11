"""
Unit tests for backend registry and format-aware selection.

This module contains tests for the backend registry, format support checks,
and the format-aware backend selection logic.
"""

# Import the backend module
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from local_read_mcp.backends.base import (
    BackendRegistry,
    BackendType,
    DocumentBackend,
    get_registry,
)


class MockSimpleBackend(DocumentBackend):
    """Mock Simple backend that supports all formats."""

    @property
    def name(self) -> str:
        return "Simple"

    @property
    def description(self) -> str:
        return "Simple backend"

    @property
    def available(self) -> bool:
        return True

    def process(self, file_path: Path, format: str, **kwargs):
        return {"format": format, "backend": "simple"}


class MockMinerUBackend(DocumentBackend):
    """Mock MinerU backend that only supports PDF."""

    @property
    def name(self) -> str:
        return "MinerU"

    @property
    def description(self) -> str:
        return "MinerU backend"

    @property
    def available(self) -> bool:
        return True

    def supports_format(self, format: str) -> bool:
        return format == "pdf"

    def process(self, file_path: Path, format: str, **kwargs):
        return {"format": format, "backend": "mineru"}


class MockOpenAIVLBackend(DocumentBackend):
    """Mock OpenAI VLM backend that supports PDF and images."""

    @property
    def name(self) -> str:
        return "OpenAI VLM"

    @property
    def description(self) -> str:
        return "OpenAI VLM backend"

    @property
    def available(self) -> bool:
        return True

    def supports_format(self, format: str) -> bool:
        return format in ["pdf", "png", "jpg", "jpeg"]

    def process(self, file_path: Path, format: str, **kwargs):
        return {"format": format, "backend": "openai-vlm"}


class TestSupportsFormat:
    """Tests for the supports_format interface."""

    def test_default_supports_all_formats(self):
        """Default implementation should support all formats."""
        backend = MockSimpleBackend()
        assert backend.supports_format("pdf") is True
        assert backend.supports_format("word") is True
        assert backend.supports_format("text") is True
        assert backend.supports_format("png") is True

    def test_mineru_only_supports_pdf(self):
        """MinerU should only support PDF."""
        backend = MockMinerUBackend()
        assert backend.supports_format("pdf") is True
        assert backend.supports_format("word") is False
        assert backend.supports_format("text") is False
        assert backend.supports_format("png") is False

    def test_vlm_supports_pdf_and_images(self):
        """VLM should support PDF and image formats."""
        backend = MockOpenAIVLBackend()
        assert backend.supports_format("pdf") is True
        assert backend.supports_format("png") is True
        assert backend.supports_format("jpg") is True
        assert backend.supports_format("jpeg") is True
        assert backend.supports_format("word") is False
        assert backend.supports_format("text") is False


class TestBackendRegistryFormatAware:
    """Tests for format-aware backend selection in BackendRegistry."""

    def setup_method(self):
        """Set up a fresh registry for each test."""
        self.registry = BackendRegistry()
        self.registry.register(BackendType.SIMPLE, MockSimpleBackend)
        self.registry.register(BackendType.MINERU, MockMinerUBackend)
        self.registry.register(BackendType.OPENAI_VLM, MockOpenAIVLBackend)

    def test_select_best_without_format_uses_priority(self):
        """Without format, select_best should use priority order."""
        # Priority: mineru -> openai-vlm -> simple
        backend = self.registry.select_best()
        assert backend.name == "MinerU"

    def test_select_best_with_pdf_prefers_mineru(self):
        """With PDF format, should prefer MinerU."""
        backend = self.registry.select_best("pdf")
        assert backend.name == "MinerU"

    def test_select_best_with_word_selects_simple(self):
        """With Word format, only Simple is available."""
        backend = self.registry.select_best("word")
        assert backend.name == "Simple"

    def test_select_best_with_text_selects_simple(self):
        """With text format, only Simple is available."""
        backend = self.registry.select_best("text")
        assert backend.name == "Simple"

    def test_select_best_with_png_selects_openai_vlm(self):
        """With PNG format, should select OpenAI VLM (MinerU doesn't support)."""
        backend = self.registry.select_best("png")
        assert backend.name == "OpenAI VLM"

    def test_select_best_with_jpg_selects_openai_vlm(self):
        """With JPG format, should select OpenAI VLM."""
        backend = self.registry.select_best("jpg")
        assert backend.name == "OpenAI VLM"

    def test_select_best_falls_back_to_simple(self):
        """With an unsupported format, should fall back to Simple."""
        backend = self.registry.select_best("unknown-format")
        assert backend.name == "Simple"


class TestBackendRegistryWithUnavailableBackends:
    """Tests for backend selection when some backends are unavailable."""

    def setup_method(self):
        """Set up a registry with some unavailable backends."""
        class UnavailableMinerU(MockMinerUBackend):
            @property
            def available(self) -> bool:
                return False

        class UnavailableVLM(MockOpenAIVLBackend):
            @property
            def available(self) -> bool:
                return False

        self.registry = BackendRegistry()
        self.registry.register(BackendType.SIMPLE, MockSimpleBackend)
        self.registry.register(BackendType.MINERU, UnavailableMinerU)
        self.registry.register(BackendType.OPENAI_VLM, UnavailableVLM)

    def test_select_best_with_unavailable_mineru(self):
        """When MinerU is unavailable, should skip to next."""
        backend = self.registry.select_best("pdf")
        assert backend.name == "Simple"


class TestRealBackendsFormatSupport:
    """Integration tests with real backend implementations."""

    def test_simple_backend_supports_all_formats(self):
        """Simple backend should support all formats."""
        registry = get_registry()
        simple = registry.get(BackendType.SIMPLE)
        assert simple is not None
        assert simple.supports_format("pdf") is True
        assert simple.supports_format("word") is True
        assert simple.supports_format("text") is True
        assert simple.supports_format("json") is True

    def test_mineru_backend_only_supports_pdf(self):
        """MinerU backend should only support PDF (if available)."""
        registry = get_registry()
        mineru = registry.get(BackendType.MINERU)
        if mineru:  # Only test if MinerU is registered
            assert mineru.supports_format("pdf") is True
            assert mineru.supports_format("word") is False
            assert mineru.supports_format("text") is False

    def test_vlm_backends_support_pdf_and_images(self):
        """VLM backends should support PDF and images (if available)."""
        registry = get_registry()

        for backend_type in [BackendType.OPENAI_VLM, BackendType.QWEN_VL]:
            backend = registry.get(backend_type)
            if backend:  # Only test if backend is registered/available
                assert backend.supports_format("pdf") is True
                assert backend.supports_format("png") is True
                assert backend.supports_format("jpg") is True
                assert backend.supports_format("jpeg") is True
                assert backend.supports_format("word") is False
                assert backend.supports_format("text") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
