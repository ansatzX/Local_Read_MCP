"""
VLM-Hybrid backend implementation using MinerU for high-quality document parsing.

Calls MinerU's hybrid-auto-engine directly (bypasses do_parse's callback + file I/O).
"""

import logging
import os
from pathlib import Path
from typing import Any

from ..intermediate_json import IntermediateJSONBuilder
from .base import DocumentBackend
from .model_detector import get_model_detector

logger = logging.getLogger(__name__)

# ── Project-local MinerU config ─────────────────────────────────────
_MINERU_CONFIG_PATH = Path(__file__).resolve().parents[3] / "mineru.json"
if _MINERU_CONFIG_PATH.exists():
    os.environ.setdefault("MINERU_TOOLS_CONFIG_JSON", str(_MINERU_CONFIG_PATH))
else:
    logger.warning("mineru.json not found at %s; MinerU will fall back to ~/mineru.json",
                   _MINERU_CONFIG_PATH)

# Try to import MinerU - optional dependency
try:
    import mineru  # noqa: F401 — ensure the package is importable
    from mineru.cli.common import read_fn
    from mineru.data.data_reader_writer import FileBasedDataWriter
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False


class VlmHybridBackend(DocumentBackend):
    """VLM-Hybrid backend using MinerU for high-quality document parsing.

    Combines a Vision Language Model for layout analysis with local pipeline
    models (OCR, formula recognition, table structure) for content extraction.
    """

    def __init__(self):
        self._detector = get_model_detector()
        self._models_initialized = False

    @property
    def name(self) -> str:
        return "VLM-Hybrid"

    @property
    def description(self) -> str:
        return ("VLM-guided document parsing with layout analysis, formula "
                "recognition, and table recognition (via MinerU)")

    @property
    def available(self) -> bool:
        if not MINERU_AVAILABLE:
            return False
        return self._detector.mineru_available

    @property
    def warning(self) -> str | None:
        if self.available:
            return None
        if not MINERU_AVAILABLE:
            return (
                "MinerU package not installed. To use VLM-Hybrid backend:\n"
                "  1. Install MinerU: pip install mineru\n"
                "  2. Download models following: "
                "https://github.com/opendatalab/MinerU#model-download\n"
                "Falling back to Simple backend."
            )
        return self._detector.mineru_warning

    def supports_format(self, format: str) -> bool:
        return format == "pdf"

    def process(
        self,
        file_path: Path,
        format: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Process a PDF using MinerU's hybrid-auto-engine.

        Directly calls ``hybrid_analyze.doc_analyze()`` which returns
        ``(middle_json, infer_result, vlm_ocr_enable)``.
        No temp directories, no ``do_parse`` callback pattern.
        """
        logger.info("VLM-Hybrid processing: %s", file_path)

        if not self.available:
            raise RuntimeError("VLM-Hybrid backend is not available")
        if format != "pdf":
            raise ValueError(f"VLM-Hybrid backend only supports PDF, got {format}")

        extract_images = kwargs.get("extract_images", False)
        images_output_dir = kwargs.get("images_output_dir")
        formula_enable = kwargs.get("formula_enable", True)
        table_enable = kwargs.get("table_enable", True)
        language = kwargs.get("language", "ch")

        try:
            # Read PDF bytes via MinerU's read_fn (handles image→PDF conversion)
            pdf_bytes = read_fn(file_path)

            # Setup image writer for extracted images
            image_writer = None
            if extract_images and images_output_dir:
                img_dir = Path(images_output_dir)
                img_dir.mkdir(parents=True, exist_ok=True)
                image_writer = FileBasedDataWriter(str(img_dir))

            # Set env flags for MinerU's VLM
            os.environ["MINERU_VLM_FORMULA_ENABLE"] = str(formula_enable).lower()
            os.environ["MINERU_VLM_TABLE_ENABLE"] = str(table_enable).lower()

            # Lazy-import hybrid analyzer (may raise if torch missing)
            from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze  # noqa: PLC0415

            # PDF classification to determine OCR mode
            from ..mineru import classify_pdf  # noqa: PLC0415
            classification = classify_pdf(pdf_bytes)
            parse_method = kwargs.get("parse_method", "auto")
            if classification == "ocr" and parse_method == "auto":
                parse_method = "ocr"

            # Call hybrid analyzer — returns directly, no callback needed
            middle_json, infer_result, vlm_ocr_enable = hybrid_doc_analyze(
                pdf_bytes=pdf_bytes,
                image_writer=image_writer,
                backend="auto-engine",
                parse_method=parse_method,
                language=language,
                inline_formula_enable=formula_enable,
            )

            logger.info("MinerU hybrid done: pages=%d, vlm_ocr=%s",
                        len(middle_json.get("pdf_info", [])), vlm_ocr_enable)

            return self._convert_mineru_to_intermediate(middle_json, file_path, format)

        except Exception as e:
            logger.warning("MinerU hybrid processing failed: %s", e)
            raise

    # ── MinerU → Intermediate JSON conversion ────────────────────────

    def _convert_mineru_to_intermediate(
        self,
        mineru_result: dict[str, Any],
        file_path: Path,
        format: str,
    ) -> dict[str, Any]:
        """Convert MinerU's middle JSON to our intermediate JSON format."""
        file_size = None
        try:
            file_size = file_path.stat().st_size
        except (FileNotFoundError, OSError):
            pass

        page_count = len(mineru_result.get("pdf_info", []))

        builder = IntermediateJSONBuilder(
            source_path=str(file_path.absolute()),
            source_format=format,
            page_count=page_count,
            file_size=file_size,
        )

        for page_info in mineru_result.get("pdf_info", []):
            page_idx = page_info.get("page_idx", 0)
            page_size = page_info.get("page_size", [612, 792])
            page_w, page_h = page_size

            for block in page_info.get("preproc_blocks", []):
                block_type = block.get("type")
                bbox = block.get("bbox", [0, 0, page_w, page_h])
                content = self._extract_block_content(block)
                if content:
                    builder.add_block(
                        type=block_type,
                        page=page_idx + 1,
                        bbox=bbox,
                        content=content,
                    )

        return builder.build()

    @staticmethod
    def _extract_block_content(block: dict[str, Any]) -> str:
        """Extract text content from a MinerU block (recursive)."""
        content_parts = []
        if "content" in block:
            content_parts.append(str(block["content"]))
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "content" in span:
                    content_parts.append(str(span["content"]))
        for sub_block in block.get("blocks", []):
            sub_content = VlmHybridBackend._extract_block_content(sub_block)
            if sub_content:
                content_parts.append(sub_content)
        return "\n".join(content_parts)
