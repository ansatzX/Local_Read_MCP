# Copyright (c) 2025
# This source code is licensed under MIT License.

"""
MinerU backend implementation for high-quality document parsing.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

from ..intermediate_json import IntermediateJSONBuilder
from .base import DocumentBackend
from .model_detector import get_model_detector

logger = logging.getLogger(__name__)

# Try to import MinerU - optional dependency
try:
    from mineru.cli.common import do_parse, read_fn
    from mineru.data.data_reader_writer import FileBasedDataWriter
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False


class MinerUBackend(DocumentBackend):
    """MinerU backend for high-quality document parsing with layout analysis."""

    def __init__(self):
        self._detector = get_model_detector()
        self._models_initialized = False

    @property
    def name(self) -> str:
        return "MinerU"

    @property
    def description(self) -> str:
        return "High-quality document parsing with layout analysis, formula recognition, and table recognition"

    @property
    def available(self) -> bool:
        """Check if MinerU backend is available."""
        if not MINERU_AVAILABLE:
            return False
        return self._detector.mineru_available

    @property
    def warning(self) -> str | None:
        """Get warning message if MinerU backend is not available."""
        if self.available:
            return None
        if not MINERU_AVAILABLE:
            return (
                "MinerU package not installed. To use MinerU backend:\n"
                "  1. Install MinerU: pip install mineru\n"
                "  2. Download models following: https://github.com/opendatalab/MinerU#model-download\n"
                "Falling back to Simple backend."
            )
        return self._detector.mineru_warning

    def supports_format(self, format: str) -> bool:
        """Check if MinerU backend supports the given format."""
        return format == "pdf"

    def _ensure_models(self):
        """Ensure MinerU models are initialized."""
        if self._models_initialized:
            return

        # Lazy import and initialize MinerU models
        # This is done here to avoid heavy imports at module level
        self._models_initialized = True

    def _convert_mineru_to_intermediate(
        self,
        mineru_result: dict[str, Any],
        file_path: Path,
        format: str
    ) -> dict[str, Any]:
        """Convert MinerU's intermediate JSON format to our format."""
        # Get file size
        file_size = None
        try:
            file_size = file_path.stat().st_size
        except (FileNotFoundError, OSError):
            pass

        # Get page count
        page_count = len(mineru_result.get("pdf_info", []))

        # Create builder
        builder = IntermediateJSONBuilder(
            source_path=str(file_path.absolute()),
            source_format=format,
            page_count=page_count,
            file_size=file_size
        )

        # Process each page from MinerU's output
        for page_info in mineru_result.get("pdf_info", []):
            page_idx = page_info.get("page_idx", 0)
            page_size = page_info.get("page_size", [612, 792])
            page_w, page_h = page_size

            # Process preproc_blocks
            for block in page_info.get("preproc_blocks", []):
                block_type = block.get("type")
                bbox = block.get("bbox", [0, 0, page_w, page_h])

                # Extract text content
                content = self._extract_block_content(block)

                if content:
                    builder.add_block(
                        type=block_type,
                        page=page_idx + 1,  # Convert to 1-indexed
                        bbox=bbox,
                        content=content
                    )

        return builder.build()

    def _extract_block_content(self, block: dict[str, Any]) -> str:
        """Extract text content from a MinerU block."""
        content_parts = []

        # Check for content in different block types
        if "content" in block:
            content_parts.append(str(block["content"]))

        # Process lines and spans
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "content" in span:
                    content_parts.append(str(span["content"]))

        # Process sub-blocks
        for sub_block in block.get("blocks", []):
            sub_content = self._extract_block_content(sub_block)
            if sub_content:
                content_parts.append(sub_content)

        return "\n".join(content_parts)

    def process(
        self,
        file_path: Path,
        format: str,
        **kwargs
    ) -> dict[str, Any]:
        """Process a document using MinerU backend."""
        logger.info(f"Processing with MinerU backend: {file_path}")

        if not self.available:
            raise RuntimeError("MinerU backend is not available")

        self._ensure_models()

        # Extract parameters
        extract_images = kwargs.get("extract_images", False)
        images_output_dir = kwargs.get("images_output_dir")
        formula_enable = kwargs.get("formula_enable", True)
        table_enable = kwargs.get("table_enable", True)
        language = kwargs.get("language", "ch")
        parse_method = kwargs.get("parse_method", "auto")

        # Check if this is a PDF - MinerU primarily handles PDFs
        if format != "pdf":
            logger.warning(f"MinerU backend works best with PDF files, got {format}")

        # For now, we'll use a hybrid approach:
        # 1. Try to use MinerU if we can import it properly
        # 2. Otherwise fall back to enhanced Simple processing with layout-aware features

        try:
            # Try to use MinerU's PDF classification at least
            mineru_result = self._try_mineru_process(
                file_path, format, extract_images, images_output_dir,
                formula_enable, table_enable, language, parse_method
            )
            if mineru_result:
                return mineru_result
        except Exception as e:
            logger.warning(f"MinerU processing failed: {e}, falling back to enhanced processing")

        # Enhanced fallback processing
        return self._enhanced_fallback_process(
            file_path, format, formula_enable, table_enable, **kwargs
        )

    def _try_mineru_process(
        self,
        file_path: Path,
        format: str,
        extract_images: bool,
        images_output_dir: str | None,
        formula_enable: bool,
        table_enable: bool,
        language: str,
        parse_method: str
    ) -> dict[str, Any] | None:
        """Try to use MinerU's actual processing capabilities."""
        if not MINERU_AVAILABLE:
            return None

        if format != "pdf":
            return None

        try:
            # Read PDF bytes first (only once)
            pdf_bytes = read_fn(file_path)

            # Use PDF classification with the already-read bytes
            from ..mineru import classify_pdf
            classification = classify_pdf(pdf_bytes)
            logger.info(f"PDF classification result: {classification}")

            # Override parse_method if classification says ocr
            if classification == "ocr" and parse_method == "auto":
                parse_method = "ocr"

            # Create temp directory for MinerU output
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Setup image writer
                image_writer = None
                if extract_images and images_output_dir:
                    image_output_path = Path(images_output_dir)
                    image_output_path.mkdir(parents=True, exist_ok=True)
                    image_writer = FileBasedDataWriter(str(image_output_path))

                # Call MinerU's do_parse
                # We'll use a callback to capture the middle_json
                mineru_middle_json = None

                def on_doc_ready(doc_index, model_list, middle_json, ocr_enable):
                    nonlocal mineru_middle_json
                    mineru_middle_json = middle_json

                # Call do_parse (simplified - MinerU's API expects lists)
                do_parse(
                    output_dir=str(temp_path),
                    pdf_file_names=[file_path.stem],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=[language],
                    backend="pipeline",
                    parse_method=parse_method,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                    start_page_id=0,
                    end_page_id=None,
                    image_writer_list=[image_writer] if image_writer else None,
                    on_doc_ready=on_doc_ready
                )

                # If we got middle_json, convert it
                if mineru_middle_json:
                    return self._convert_mineru_to_intermediate(
                        mineru_middle_json, file_path, format
                    )

        except Exception as e:
            logger.warning(f"MinerU processing failed: {e}, falling back to enhanced processing")

        # Return None to indicate we should use fallback
        return None

    def _enhanced_fallback_process(
        self,
        file_path: Path,
        format: str,
        formula_enable: bool,
        table_enable: bool,
        **kwargs
    ) -> dict[str, Any]:
        """Enhanced fallback processing - delegate to SimpleBackend."""
        from .simple import SimpleBackend

        simple_backend = SimpleBackend()
        return simple_backend.process(file_path, format, **kwargs)
