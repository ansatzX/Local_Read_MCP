"""
Unit tests for MCP server tools.

This module contains tests for the FastMCP server implementation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from local_read_mcp.server import app as server_app


class TestProcessBinaryFileExtractImagesDefault:
    """Tests for extract_images default behavior in process_binary_file."""

    @pytest.mark.asyncio
    async def test_auto_enable_extract_images_when_vision_enabled(self, monkeypatch, tmp_path):
        """When not specified and vision is enabled, processing should extract images for PDFs."""
        called = {}
        test_file = tmp_path / "sample.pdf"
        test_file.write_text("fake pdf", encoding="utf-8")

        class FakeBackend:
            name = "Fake"
            warning = None

            def supports_format(self, format_name):
                return True

            def process(self, file_path, format_name, **kwargs):
                called.update(kwargs)
                return {
                    "source": {"path": str(file_path), "format": format_name, "page_count": 1},
                    "metadata": {},
                    "blocks": {
                        "block_00000000": {
                            "type": "text",
                            "page": 1,
                            "bbox": [0, 0, 612, 792],
                            "confidence": 0.9,
                            "content": "content",
                        }
                    },
                    "reading_order": ["block_00000000"],
                }

        class FakeRegistry:
            def select_best(self, format_name=None):
                return FakeBackend()

            def get(self, backend_type):
                return FakeBackend()

        monkeypatch.setattr(server_app, "get_registry", lambda: FakeRegistry())
        monkeypatch.setattr(server_app, "VISION_ENABLED", True)

        result = await server_app.process_binary_file.fn(
            file_path=str(test_file),
            format="pdf",
        )

        assert result["success"] is True
        assert called["extract_images"] is True

    @pytest.mark.asyncio
    async def test_default_extract_images_false_when_vision_disabled(self, monkeypatch, tmp_path):
        """When vision is disabled and not specified, processing should not extract images."""
        called = {}
        test_file = tmp_path / "sample.pdf"
        test_file.write_text("fake pdf", encoding="utf-8")

        class FakeBackend:
            name = "Fake"
            warning = None

            def supports_format(self, format_name):
                return True

            def process(self, file_path, format_name, **kwargs):
                called.update(kwargs)
                return {
                    "source": {"path": str(file_path), "format": format_name, "page_count": 1},
                    "metadata": {},
                    "blocks": {
                        "block_00000000": {
                            "type": "text",
                            "page": 1,
                            "bbox": [0, 0, 612, 792],
                            "confidence": 0.9,
                            "content": "content",
                        }
                    },
                    "reading_order": ["block_00000000"],
                }

        class FakeRegistry:
            def select_best(self, format_name=None):
                return FakeBackend()

            def get(self, backend_type):
                return FakeBackend()

        monkeypatch.setattr(server_app, "get_registry", lambda: FakeRegistry())
        monkeypatch.setattr(server_app, "VISION_ENABLED", False)

        result = await server_app.process_binary_file.fn(
            file_path=str(test_file),
            format="pdf",
        )

        assert result["success"] is True
        assert called["extract_images"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestProcessBinaryFileFallback:
    """Tests for VLM-to-Simple fallback in process_binary_file."""

    @pytest.mark.asyncio
    async def test_process_binary_file_falls_back_to_simple_when_vlm_process_fails(self, monkeypatch, tmp_path):
        from local_read_mcp.server import app as server_app
        from local_read_mcp.backends.base import BackendType

        test_file = tmp_path / 'sample.pdf'
        test_file.write_text('fake pdf', encoding='utf-8')

        class FailingVlm:
            name = 'VLM-Hybrid'
            warning = None
            def supports_format(self, format_name): return format_name == 'pdf'
            def process(self, file_path, format_name, **kwargs):
                raise RuntimeError('mineru failed')

        class PassingSimple:
            name = 'Simple'
            warning = None
            def supports_format(self, format_name): return True
            def process(self, file_path, format_name, **kwargs):
                return {
                    'source': {'path': str(file_path), 'format': format_name, 'page_count': 1},
                    'metadata': {},
                    'blocks': {'block_00000000': {'type': 'text', 'page': 1, 'bbox': [0, 0, 1, 1], 'content': 'ok'}},
                    'reading_order': ['block_00000000'],
                }

        class FakeRegistry:
            def select_best(self, format_name=None): return FailingVlm()
            def get(self, backend_type):
                if backend_type == BackendType.SIMPLE:
                    return PassingSimple()
                return FailingVlm()

        monkeypatch.setattr(server_app, 'get_registry', lambda: FakeRegistry())

        result = await server_app.process_binary_file.fn(file_path=str(test_file), format='pdf')

        assert result['success'] is True
        assert result['backend_used'] == 'Simple'
        assert any('VLM-Hybrid' in warning and 'mineru failed' in warning for warning in result['warnings'])



class TestProcessBinaryFileChunkPlanning:
    """Tests for chunk parameter semantics."""

    def test_plan_chunks_uses_page_batch_size_for_fixed_auto(self, monkeypatch):
        from local_read_mcp.server import orchestrator
        from types import SimpleNamespace

        class FakeDoc:
            page_count = 50
            def get_toc(self): return []
            def __getitem__(self, index):
                class Page:
                    def get_text(self): return ''
                    def get_label(self): return ''
                return Page()
            def close(self): pass

        monkeypatch.setitem(__import__('sys').modules, 'fitz', SimpleNamespace(open=lambda path: FakeDoc()))

        chunks = orchestrator.plan_chunks(
            file_path='sample.pdf',
            format='pdf',
            backend_name='Simple',
            chapter_split='auto',
            start_page=None,
            end_page=None,
            page_batch_size=7,
        )

        assert [c.phys_end - c.phys_start + 1 for c in chunks[:2]] == [7, 7]

    def test_plan_chunks_true_is_not_treated_as_one_page_chunks(self, monkeypatch):
        from local_read_mcp.server import orchestrator
        from types import SimpleNamespace

        class FakeDoc:
            page_count = 50
            def get_toc(self): return []
            def __getitem__(self, index):
                class Page:
                    def get_text(self): return ''
                    def get_label(self): return ''
                return Page()
            def close(self): pass

        monkeypatch.setitem(__import__('sys').modules, 'fitz', SimpleNamespace(open=lambda path: FakeDoc()))

        chunks = orchestrator.plan_chunks(
            file_path='sample.pdf',
            format='pdf',
            backend_name='Simple',
            chapter_split=True,
            start_page=None,
            end_page=None,
            page_batch_size=10,
        )

        assert all((c.phys_end - c.phys_start + 1) > 1 for c in chunks)

    def test_plan_chunks_page_batch_size_defaults_to_64(self, monkeypatch):
        from local_read_mcp.server import orchestrator
        from types import SimpleNamespace

        class FakeDoc:
            page_count = 50
            def get_toc(self): return []
            def __getitem__(self, index):
                class Page:
                    def get_text(self): return ''
                    def get_label(self): return ''
                return Page()
            def close(self): pass

        monkeypatch.setitem(__import__('sys').modules, 'fitz', SimpleNamespace(open=lambda path: FakeDoc()))

        # chapter_split='auto' on 50 pages -> splits (50>30), no TOC, uses page_batch_size=64 (1 chunk)
        chunks = orchestrator.plan_chunks(
            file_path='sample.pdf',
            format='pdf',
            backend_name='Simple',
            chapter_split='auto',
            start_page=None,
            end_page=None,
            page_batch_size=64,
        )

        assert len(chunks) == 1
        assert chunks[0].phys_end - chunks[0].phys_start + 1 == 50



class TestProcessBinaryFileMultiChunk:
    """Tests for multi-chunk processing output shape."""

    @pytest.mark.asyncio
    async def test_multi_chunk_writes_top_level_artifacts(self, monkeypatch, tmp_path):
        from local_read_mcp.server import app as server_app
        from local_read_mcp.segmenter import Chunk

        test_file = tmp_path / 'sample.pdf'
        test_file.write_text('fake pdf', encoding='utf-8')

        monkeypatch.chdir(tmp_path)

        class FakeBackend:
            name = 'Simple'
            warning = None
            def supports_format(self, format_name): return True
            def process(self, file_path, format_name, **kwargs):
                return {
                    'source': {'path': str(file_path), 'format': format_name, 'page_count': 1},
                    'metadata': {},
                    'blocks': {'block_00000000': {'type': 'text', 'page': 0, 'bbox': [0, 0, 1, 1], 'content': 'ok'}},
                    'reading_order': ['block_00000000'],
                }

        class FakeRegistry:
            def select_best(self, format_name=None): return FakeBackend()
            def get(self, backend_type): return FakeBackend()

        monkeypatch.setattr(server_app, 'get_registry', lambda: FakeRegistry())
        monkeypatch.setattr(server_app, 'plan_chunks', lambda **kwargs: [
            Chunk(phys_start=0, phys_end=0, title='a'),
            Chunk(phys_start=1, phys_end=1, title='b'),
        ])

        result = await server_app.process_binary_file.fn(file_path=str(test_file), format='pdf')

        assert result['success'] is True
        assert Path(result['files']['intermediate_json']).exists()
        assert Path(result['files']['markdown']).exists()
        assert Path(result['files']['index_json']).exists()
        assert Path(result['files']['structural_toc']).exists()
