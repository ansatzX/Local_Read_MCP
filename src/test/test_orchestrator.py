"""Unit tests for server orchestrator helpers."""

from pathlib import Path
from types import SimpleNamespace


def test_process_and_save_slices_pdf_chunk_without_name_error(monkeypatch, tmp_path):
    """PDF chunk slicing path should work (regression for missing tempfile import)."""
    from local_read_mcp.server import orchestrator
    from local_read_mcp.segmenter import Chunk

    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_text("fake pdf", encoding="utf-8")

    class FakeSrcDoc:
        page_count = 5

        def close(self):
            return None

    class FakeSlicedDoc:
        def insert_pdf(self, src, from_page, to_page):
            self.from_page = from_page
            self.to_page = to_page

        def save(self, path):
            Path(path).write_text("sliced", encoding="utf-8")

        def close(self):
            return None

    def fake_fitz_open(*args, **kwargs):
        if len(args) == 0:
            return FakeSlicedDoc()
        return FakeSrcDoc()

    monkeypatch.setitem(
        __import__("sys").modules,
        "fitz",
        SimpleNamespace(open=fake_fitz_open),
    )

    class FakeBackend:
        def process(self, file_path, format_name, **kwargs):
            return {
                "source": {"path": str(file_path), "format": format_name, "page_count": 1},
                "metadata": {},
                "blocks": {
                    "block_00000000": {
                        "type": "text",
                        "page": 0,
                        "bbox": [0, 0, 1, 1],
                        "content": "ok",
                    }
                },
                "reading_order": ["block_00000000"],
            }

    output_path = tmp_path / "out"
    output_path.mkdir()
    images_dir = output_path / "images"
    chunk = Chunk(phys_start=1, phys_end=2, title="middle")

    result = orchestrator.process_and_save(
        file_path=str(input_pdf),
        backend=FakeBackend(),
        format="pdf",
        output_path=output_path,
        images_dir=images_dir,
        chunk=chunk,
        backend_kwargs={},
    )

    assert "sliced_pdf_path" in result
    assert Path(result["sliced_pdf_path"]).exists()

