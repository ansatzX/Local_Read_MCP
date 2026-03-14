import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from .base import (
    DocumentConverterResult,
    pdfminer,
    fitz
)
from .utils import extract_sections_from_markdown, fix_latex_formulas, apply_content_limit


def extract_pdf_images(
    pdf_path: str,
    output_dir: Optional[str] = None,
    page_range: Optional[tuple] = None
) -> List[Dict[str, Any]]:
    """
    Extract images from PDF file using PyMuPDF.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images. If None, creates temp directory
        page_range: Tuple of (start_page, end_page) to extract from specific pages.
                    If None, extracts from all pages. Page numbers are 0-indexed.

    Returns:
        List of dictionaries containing image information:
        [
            {
                "page": int,  # Page number (0-indexed)
                "index": int,  # Image index on the page
                "xref": int,  # PDF object reference
                "width": int,  # Image width in pixels
                "height": int,  # Image height in pixels
                "format": str,  # Image format (png, jpeg, etc.)
                "size": int,  # File size in bytes
                "saved_path": str,  # Path where image was saved
            },
            ...
        ]

    Raises:
        ImportError: If PyMuPDF (fitz) is not installed
        FileNotFoundError: If PDF file doesn't exist
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF (fitz) is required for image extraction. "
            "Install with: pip install pymupdf"
        )

    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="pdf_images_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Extracting images from PDF: {pdf_path}")
    logger.info(f"Saving images to: {output_dir}")

    images_info = []

    try:
        doc = fitz.open(pdf_path)

        # Determine page range
        start_page = 0
        end_page = len(doc)
        if page_range:
            start_page, end_page = page_range
            start_page = max(0, start_page)
            end_page = min(len(doc), end_page)

        # Extract images from each page
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            image_list = page.get_images()

            logger.debug(f"Page {page_num}: found {len(image_list)} images")

            for img_index, img in enumerate(image_list):
                xref = img[0]  # XREF number

                # Extract image
                try:
                    base_image = doc.extract_image(xref)

                    # Get image properties
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]  # png, jpeg, etc.
                    image_width = base_image["width"]
                    image_height = base_image["height"]

                    # Generate filename
                    image_filename = f"page{page_num:03d}_img{img_index:02d}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)

                    # Save image
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # Record image info
                    images_info.append({
                        "page": page_num,
                        "index": img_index,
                        "xref": xref,
                        "width": image_width,
                        "height": image_height,
                        "format": image_ext,
                        "size": len(image_bytes),
                        "saved_path": image_path,
                    })

                    logger.debug(
                        f"Extracted: {image_filename} "
                        f"({image_width}x{image_height}, {len(image_bytes)} bytes)"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to extract image {img_index} from page {page_num}: {e}"
                    )
                    continue

        doc.close()

        logger.info(f"Successfully extracted {len(images_info)} images")

    except Exception as e:
        logger.error(f"Error extracting images from PDF: {e}")
        raise

    return images_info


def PdfConverter(
    local_path: str,
    extract_metadata: bool = False,
    extract_sections: bool = False,
    extract_images: bool = False,
    images_output_dir: Optional[str] = None,
    fix_latex: bool = True
) -> DocumentConverterResult:
    """
    Convert a PDF file to text format with enhanced features.

    Args:
        local_path: Path to PDF file to convert.
        extract_metadata: Whether to extract metadata (file size, etc.)
        extract_sections: Whether to extract sections from content
        extract_images: Whether to extract images from PDF (requires PyMuPDF)
        images_output_dir: Directory to save extracted images (default: temp directory)
        fix_latex: Whether to fix LaTeX formula parsing issues

    Returns:
        DocumentConverterResult containing extracted text and optional metadata/sections/images.
    """
    logger = logging.getLogger(__name__)

    if pdfminer is None:
        return DocumentConverterResult(
            title=None,
            text_content="[Error: pdfminer-six not installed]",
            error="pdfminer-six not installed"
        )

    try:
        # Extract text content
        text_content = pdfminer.high_level.extract_text(local_path)

        # Get PDF page count
        pdf_page_count = None
        try:
            from pdfminer.pdfparser import PDFParser
            from pdfminer.pdfdocument import PDFDocument
            from pdfminer.pdfpage import PDFPage

            with open(local_path, 'rb') as fp:
                parser = PDFParser(fp)
                document = PDFDocument(parser)
                pdf_page_count = len(list(PDFPage.create_pages(document)))
        except Exception as e:
            logger.warning(f"Could not extract PDF page count: {e}")

        # Apply LaTeX fixes if requested
        if fix_latex:
            text_content = fix_latex_formulas(text_content)

        # Apply content limit (200,000 characters)
        text_content = apply_content_limit(text_content)

        # Prepare metadata
        metadata = {}
        sections = []
        images = []

        if extract_metadata:
            file_size = None
            try:
                file_size = os.path.getsize(local_path)
            except (OSError, Exception):
                pass
            metadata = {
                "file_path": local_path,
                "file_size": file_size,
                "file_extension": os.path.splitext(local_path)[1],
                "conversion_timestamp": time.time(),
                "pdf_page_count": pdf_page_count,  # Add actual PDF page count
            }

        if extract_sections:
            sections = extract_sections_from_markdown(text_content)

        # Extract images if requested
        if extract_images:
            if fitz is None:
                logger.warning(
                    "PyMuPDF not installed. Cannot extract images. "
                    "Install with: pip install pymupdf"
                )
                metadata["image_extraction_error"] = "PyMuPDF not installed"
            else:
                try:
                    images = extract_pdf_images(local_path, output_dir=images_output_dir)
                    logger.info(f"Extracted {len(images)} images from PDF")

                    # Add image count to metadata
                    if extract_metadata:
                        metadata["image_count"] = len(images)
                        if images_output_dir:
                            metadata["images_directory"] = images_output_dir

                except Exception as e:
                    logger.error(f"Failed to extract images: {e}")
                    metadata["image_extraction_error"] = str(e)

        # Try to extract title from first line or metadata
        title = None
        if text_content:
            first_line = text_content.split('\n')[0].strip()
            if first_line and len(first_line) < 200:  # Reasonable title length
                title = first_line

        return DocumentConverterResult(
            title=title,
            text_content=text_content,
            metadata=metadata,
            sections=sections,
            tables=[],  # PDF tables not extracted in basic version
            images=images,  # Extracted images
            processing_time_ms=None  # Can be calculated by caller
        )

    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        return DocumentConverterResult(
            title=None,
            text_content=f"Error converting PDF: {str(e)}",
            error=str(e)
        )
