################################################################################
# Local Read MCP - Document Converters
# Extracted from MiroThinker, excluding OpenAI dependencies
################################################################################

import json
import os
import re
import shutil
import tempfile
import traceback
import csv as csv_module
from typing import Any, Union
from urllib.parse import quote, unquote, urlparse, urlunparse

# Optional imports
try:
    import mammoth
except ImportError:
    mammoth = None

try:
    import markdownify
except ImportError:
    markdownify = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import pdfminer
    import pdfminer.high_level
except ImportError:
    pdfminer = None

try:
    import pptx
except ImportError:
    pptx = None

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import html
except ImportError:
    html = None

# Media file extension constants
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
AUDIO_EXTENSIONS = {"wav", "mp3", "m4a"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

class DocumentConverterResult:
    """The result of converting a document to text."""

    def __init__(self, title: Union[str, None] = None, text_content: str = ""):
        self.title: Union[str, None] = title
        self.text_content: str = text_content

class _CustomMarkdownify(markdownify.MarkdownConverter):
    """
    A custom version of markdownify's MarkdownConverter. Changes include:

    - Altering the default heading style to use '#', '##', etc.
    - Removing javascript hyperlinks.
    - Truncating images with large data:uri sources.
    - Ensuring URIs are properly escaped, and do not conflict with Markdown syntax
    """

    def __init__(self, **options: Any):
        options["heading_style"] = options.get("heading_style", markdownify.ATX)
        # Explicitly cast options to the expected type if necessary
        super().__init__(**options)

    def convert_hn(self, n: int, el: Any, text: str, convert_as_inline: bool) -> str:
        """Same as usual, but be sure to start with a new line"""
        if not convert_as_inline:
            if not re.search(r"^\n", text):
                return "\n" + super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

        return super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

    def convert_a(self, el: Any, text: str, convert_as_inline: bool):
        """Same as usual converter, but removes Javascript links and escapes URIs."""
        prefix, suffix, text = markdownify.chomp(text)  # type: ignore
        if not text:
            return ""
        href = el.get("href")
        title = el.get("title")

        # Escape URIs and skip non-http or file schemes
        if href:
            try:
                parsed_url = urlparse(href)  # type: ignore
                if parsed_url.scheme and parsed_url.scheme.lower() not in [
                    "http",
                    "https",
                    "file",
                ]:  # type: ignore
                    return "%s%s%s" % (prefix, text, suffix)
                href = urlunparse(
                    parsed_url._replace(path=quote(unquote(parsed_url.path)))
                )  # type: ignore
            except ValueError:  # It's not clear if this ever gets thrown
                return "%s%s%s" % (prefix, text, suffix)

        # For the replacement see #29: text nodes underscores are escaped
        if (
            self.options["autolinks"]
            and text.replace(r"\_", "_") == href
            and not title
            and not self.options["default_title"]
        ):
            # Shortcut syntax
            return "<%s>" % href
        if self.options["default_title"] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        return (
            "%s[%s](%s%s)%s" % (prefix, text, href, title_part, suffix)
            if href
            else text
        )

    def convert_img(self, el: Any, text: str, convert_as_inline: bool) -> str:
        """Same as usual converter, but removes data URIs"""

        alt = el.attrs.get("alt", None) or ""
        src = el.attrs.get("src", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        if (
            convert_as_inline
            and el.parent.name not in self.options["keep_inline_images_in"]
        ):
            return alt

        # Remove dataURIs
        if src.startswith("data:"):
            src = src.split(",")[0] + "..."

        return "![%s](%s%s)" % (alt, src, title_part)

    def convert_soup(self, soup: Any) -> str:
        return super().convert_soup(soup)  # type: ignore


class DocumentConverterResult:
    """The result of converting a document to text."""

    def __init__(self, title: Union[str, None] = None, text_content: str = ""):
        self.title: Union[str, None] = title
        self.text_content: str = text_content


def convert_html_to_md(html_content):
    """
    Placeholder for HTML to Markdown conversion function
    In the original class, this would call self._convert()
    """
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()

    # Print only the main content
    body_elm = soup.find("body")
    webpage_text = ""
    if body_elm:
        webpage_text = _CustomMarkdownify().convert_soup(body_elm)
    else:
        webpage_text = _CustomMarkdownify().convert_soup(soup)

    assert isinstance(webpage_text, str)

    return DocumentConverterResult(
        title=None if soup.title is None else soup.title.string,
        text_content=webpage_text,
    )


def HtmlConverter(local_path: str):
    """
    Convert an HTML file to Markdown format.

    Args:
        local_path: Path to the HTML file to convert.

    Returns:
        DocumentConverterResult containing the converted Markdown text.
    """
    with open(local_path, "rt", encoding="utf-8") as fh:
        html_content = fh.read()

        return convert_html_to_md(html_content)


def DocxConverter(local_path: str):
    """
    Convert a DOCX file to Markdown format.

    Uses mammoth library to first convert DOCX to HTML, then converts
    the HTML to Markdown.

    Args:
        local_path: Path to the DOCX file to convert.

    Returns:
        DocumentConverterResult containing the converted Markdown text.
    """
    with open(local_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html_content = result.value
    return convert_html_to_md(html_content)


def XlsxConverter(local_path: str):
    """
    Converts Excel files to Markdown using openpyxl.
    Preserves color formatting and other cell styling information.

    Args:
        local_path: Path to the Excel file

    Returns:
        DocumentConverterResult with the Markdown representation of the Excel file
    """
    # Load the workbook
    wb = openpyxl.load_workbook(local_path, data_only=True)
    md_content = ""

    # Helper function to convert RGB color to hex
    def rgb_to_hex(rgb_value):
        if not rgb_value:
            return None

        # Convert RGB value to string for processing
        rgb_string = str(rgb_value)

        # Handle RGB format like 'RGB(255, 255, 255)'
        if isinstance(rgb_value, str) and rgb_string.startswith("RGB"):
            rgb_match = re.match(r"RGB\((\d+), (\d+), (\d+)\)", rgb_string)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                return f"#{r:02x}{g:02x}{b:02x}"

        # Special handling for FFFFFFFF (white) and 00000000 (transparent/none)
        if rgb_string in ["FFFFFFFF", "00000000", "none", "auto"]:
            return None

        # Handle ARGB format (common in openpyxl)
        if len(rgb_string) == 8:  # ARGB format like 'FF5733FF'
            return f"#{rgb_string[2:]}"  # Strip alpha channel

        # Handle direct hex values like 'FF5733'
        if isinstance(rgb_value, str):
            return f"#{rgb_string}" if not rgb_string.startswith("#") else rgb_string

        return None  # Return None for unrecognized formats

    # Helper function to detect and format cell styling
    def get_cell_format_info(cell):
        info = {}

        # Get background color if it exists
        if cell.fill and hasattr(cell.fill, "fgColor") and cell.fill.fgColor:
            # Get the RGB value - in openpyxl this can be stored in different attributes
            rgb_value = None
            if hasattr(cell.fill.fgColor, "rgb") and cell.fill.fgColor.rgb:
                rgb_value = cell.fill.fgColor.rgb
            elif hasattr(cell.fill.fgColor, "value") and cell.fill.fgColor.value:
                rgb_value = cell.fill.fgColor.value

            if rgb_value:
                bg_color = rgb_to_hex(rgb_value)
                if bg_color:  # Skip transparent or white (handled in rgb_to_hex)
                    info["bg_color"] = bg_color

        # Get font color if it exists
        if cell.font and hasattr(cell.font, "color") and cell.font.color:
            # Get the RGB value - in openpyxl this can be stored in different attributes
            rgb_value = None
            if hasattr(cell.font.color, "rgb") and cell.font.color.rgb:
                rgb_value = cell.font.color.rgb
            elif hasattr(cell.font.color, "value") and cell.font.color.value:
                rgb_value = cell.font.color.value

            if rgb_value:
                font_color = rgb_to_hex(rgb_value)
                if font_color:  # Skip transparent (handled in rgb_to_hex)
                    info["font_color"] = font_color

        # Get font weight (bold)
        if cell.font and cell.font.bold:
            info["bold"] = True

        # Get font style (italic)
        if cell.font and cell.font.italic:
            info["italic"] = True

        # Get font underline
        if cell.font and cell.font.underline and cell.font.underline != "none":
            info["underline"] = True

        return info

    # Process each sheet in the workbook
    for sheet_name in wb.sheetnames:
        try:
            sheet = wb[sheet_name]
            md_content += f"## {sheet_name}\n\n"

            # Get the dimensions of the used part of the sheet
            min_row, min_col = 1, 1
            max_row = max(
                (cell.row for cell in sheet._cells.values() if cell.value is not None),
                default=0,
            )
            max_col = max(
                (
                    cell.column
                    for cell in sheet._cells.values()
                    if cell.value is not None
                ),
                default=0,
            )

            if max_row == 0 or max_col == 0:
                md_content += "This sheet is empty.\n\n"
                continue
        except Exception as e:
            error_msg = f"Error processing sheet '{sheet_name}': {str(e)}"
            print(error_msg)
            md_content += (
                f"## {sheet_name}\n\nError processing this sheet: {str(e)}\n\n"
            )
            continue

        try:
            # First, determine column widths
            col_widths = {}
            for col_idx in range(min_col, max_col + 1):
                max_length = 0
                # col_letter = get_column_letter(col_idx)
                _ = get_column_letter(col_idx)
                for row_idx in range(min_row, max_row + 1):
                    try:
                        cell = sheet.cell(row=row_idx, column=col_idx)
                        cell_value = str(cell.value) if cell.value is not None else ""
                        max_length = max(max_length, len(cell_value))
                    except Exception as e:
                        print(
                            f"Warning: Error processing cell at row {row_idx}, column {col_idx}: {str(e)}"
                        )
                        max_length = max(max_length, 10)  # Use reasonable default
                col_widths[col_idx] = max(max_length + 2, 5)  # Min width of 5

            # Start building the table
            # Header row with column separators
            md_content += "|"
            for col_idx in range(min_col, max_col + 1):
                md_content += " " + " " * col_widths[col_idx] + " |"
            md_content += "\n"

            # Separator row
            md_content += "|"
            for col_idx in range(min_col, max_col + 1):
                md_content += ":" + "-" * col_widths[col_idx] + ":|"
            md_content += "\n"

            # Data rows
            for row_idx in range(min_row, max_row + 1):
                md_content += "|"
                for col_idx in range(min_col, max_col + 1):
                    try:
                        cell = sheet.cell(row=row_idx, column=col_idx)
                        cell_value = str(cell.value) if cell.value is not None else ""

                        # Get formatting info
                        try:
                            format_info = get_cell_format_info(cell)
                        except Exception as e:
                            print(
                                f"Warning: Error getting formatting for cell at row {row_idx}, column {col_idx}: {str(e)}"
                            )
                            format_info = {}

                        formatted_value = cell_value

                        # Add HTML-style formatting if needed
                        if format_info:
                            style_parts = []

                            if "bg_color" in format_info:
                                style_parts.append(
                                    f"background-color:{format_info['bg_color']}"
                                )

                            if "font_color" in format_info:
                                style_parts.append(f"color:{format_info['font_color']}")

                            span_attributes = []
                            if style_parts:
                                span_attributes.append(
                                    f'style="{"; ".join(style_parts)}"'
                                )

                            # Format with bold/italic/underline if needed
                            inner_value = cell_value
                            if "bold" in format_info:
                                inner_value = f"<strong>{inner_value}</strong>"
                            if "italic" in format_info:
                                inner_value = f"<em>{inner_value}</em>"
                            if "underline" in format_info:
                                inner_value = f"<u>{inner_value}</u>"

                            # Only add a span if we have style attributes
                            if span_attributes:
                                formatted_value = f"<span {' '.join(span_attributes)}>{inner_value}</span>"
                            else:
                                formatted_value = inner_value

                        # Pad to column width and add to markdown
                        padding = col_widths[col_idx] - len(cell_value)
                        padded_value = " " + formatted_value + " " * (padding + 1)
                        md_content += padded_value + "|"
                    except Exception as e:
                        print(
                            f"Error processing cell at row {row_idx}, column {col_idx}: {str(e)}"
                        )
                        # Add a placeholder for the failed cell
                        padded_value = " [Error] " + " " * (col_widths[col_idx] - 7)
                        md_content += padded_value + " |"

                md_content += "\n"
        except Exception as e:
            error_msg = f"Error generating table for sheet '{sheet_name}': {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            md_content += f"Error generating table: {str(e)}\n\n"

        # Add formatting legend
        has_formatting = False
        for row_idx in range(min_row, max_row + 1):
            for col_idx in range(min_col, max_col + 1):
                cell = sheet.cell(row=row_idx, column=col_idx)
                if get_cell_format_info(cell):
                    has_formatting = True
                    break
            if has_formatting:
                break

        if has_formatting:
            md_content += "\n### Formatting Information\n"
            md_content += "The table above includes HTML formatting to represent colors and styles from the original Excel file.\n"
            md_content += "This formatting may not display in all Markdown viewers.\n"

        md_content += "\n\n"  # Extra newlines between sheets

    return DocumentConverterResult(
        title=None,
        text_content=md_content.strip(),
    )


def PptxConverter(local_path: str) -> DocumentConverterResult:
    """
    Converts PPTX files to Markdown. Supports headings, tables and images with alt text.

    Args:
        local_path: Path to the PPTX file

    Returns:
        DocumentConverterResult containing the converted Markdown text
    """

    def is_picture(shape):
        """Check if a shape is a picture"""
        if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE:
            return True
        if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PLACEHOLDER:
            if hasattr(shape, "image"):
                return True
        return False

    def is_table(shape):
        """Check if a shape is a table"""
        if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.TABLE:
            return True
        return False

    if not local_path.endswith(".pptx"):
        return DocumentConverterResult(
            title=None,
            text_content=f"Error: Expected .pptx file, got: {local_path}",
        )

    md_content = ""
    presentation = pptx.Presentation(local_path)
    slide_num = 0

    for slide in presentation.slides:
        slide_num += 1
        md_content += f"\n\n<!-- Slide number: {slide_num} -->\n"
        title = slide.shapes.title

        for shape in slide.shapes:
            # Pictures
            if is_picture(shape):
                # https://github.com/scanny/python-pptx/pull/512#issuecomment-1713100069
                alt_text = ""
                try:
                    alt_text = shape._element._nvXxPr.cNvPr.attrib.get("descr", "")
                except Exception:
                    pass
                # A placeholder name
                filename = re.sub(r"\W", "", shape.name) + ".jpg"
                md_content += (
                    "\n!["
                    + (alt_text if alt_text else shape.name)
                    + "]("
                    + filename
                    + ")\n"
                )

            # Tables
            if is_table(shape):
                html_table = "<html><body><table>"
                first_row = True
                for row in shape.table.rows:
                    html_table += "<tr>"
                    for cell in row.cells:
                        if first_row:
                            html_table += "<th>" + html.escape(cell.text) + "</th>"
                        else:
                            html_table += "<td>" + html.escape(cell.text) + "</td>"
                    html_table += "</tr>"
                    first_row = False
                html_table += "</table></body></html>"

                # Note: This would require a separate HTML to Markdown converter function
                # In this version, I'm assuming a convert_html_to_md function exists
                md_content += (
                    "\n" + convert_html_to_md(html_table).text_content.strip() + "\n"
                )

            # Text areas
            elif shape.has_text_frame:
                if shape == title:
                    md_content += "# " + shape.text.lstrip() + "\n"
                else:
                    md_content += shape.text + "\n"

        md_content = md_content.strip()
        if slide.has_notes_slide:
            md_content += "\n\n### Notes:\n"
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame is not None:
                md_content += notes_frame.text
            md_content = md_content.strip()

    return DocumentConverterResult(
        title=None,
        text_content=md_content.strip(),
    )


def ZipConverter(local_path: str, **kwargs):
    """
    Extracts ZIP files to a temporary directory and processes each file according to its extension.
    Returns a combined result of all processed files.
    """
    import zipfile

    temp_dir = tempfile.mkdtemp(prefix="zip_extract_")
    md_content = f"# Extracted from ZIP: {os.path.basename(local_path)}\n\n"

    try:
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Get all extracted files
        extracted_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, temp_dir)
                extracted_files.append((file_path, rel_path))

        if not extracted_files:
            md_content += "The ZIP file is empty or contains no files.\n"
        else:
            md_content += f"Total files extracted: {len(extracted_files)}\n\n"

            for file_path, rel_path in extracted_files:
                md_content += f"## File: {rel_path}\n\n"

                # Process each file based on its extension
                file_extension = (
                    file_path.rsplit(".", maxsplit=1)[-1].lower()
                    if "." in file_path
                    else ""
                )
                file_result = None

                try:
                    # Use the same processing logic as process_input
                    if file_extension == "py":
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_result = DocumentConverterResult(
                                title=None, text_content=f.read()
                            )

                    elif file_extension in [
                        "txt",
                        "md",
                        "sh",
                        "yaml",
                        "yml",
                        "toml",
                        "csv",
                    ]:
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_result = DocumentConverterResult(
                                title=None, text_content=f.read()
                            )

                    elif file_extension in ["jsonld", "json"]:
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_result = DocumentConverterResult(
                                title=None,
                                text_content=json.dumps(
                                    json.load(f), ensure_ascii=False, indent=2
                                ),
                            )

                    elif file_extension in ["xlsx", "xls"]:
                        file_result = XlsxConverter(local_path=file_path)

                    elif file_extension == "pdf":
                        file_result = DocumentConverterResult(
                            title=None,
                            text_content=pdfminer.high_level.extract_text(file_path),
                        )

                    elif file_extension in ["docx", "doc"]:
                        file_result = DocxConverter(local_path=file_path)

                    elif file_extension in ["html", "htm"]:
                        file_result = HtmlConverter(local_path=file_path)

                    elif file_extension in ["pptx", "ppt"]:
                        file_result = PptxConverter(local_path=file_path)

                    elif file_extension in IMAGE_EXTENSIONS:
                        # Media files noted but not processed (no external API for captions)
                        md_content += f"[{file_extension.upper()} file - processing not available without external API]\n\n"
                        continue

                    elif file_extension in AUDIO_EXTENSIONS:
                        # Media files noted but not processed (no external API for captions)
                        md_content += f"[{file_extension.upper()} file - processing not available without external API]\n\n"
                        continue

                    elif file_extension in VIDEO_EXTENSIONS:
                        # Media files noted but not processed (no external API for captions)
                        md_content += f"[{file_extension.upper()} file - processing not available without external API]\n\n"
                        continue

                    elif file_extension == "pdb":
                        md_content += "[PDB file - specialized format]\n\n"
                        continue

                    else:
                        # Try MarkItDown as fallback
                        try:
                            md_tool = MarkItDown(enable_plugins=True)
                            file_result = md_tool.convert(file_path)
                        except Exception:
                            md_content += (
                                f"[Unsupported file type: {file_extension}]\n\n"
                            )
                            continue

                    # Add the processed content
                    if file_result and getattr(file_result, "text_content", None):
                        content = file_result.text_content
                        # Limit length for each file
                        max_len = 50_000
                        if len(content) > max_len:
                            content = content[:max_len] + "\n... [Content truncated]"
                        md_content += f"```\n{content}\n```\n\n"

                except Exception as e:
                    md_content += f"[Error processing file: {str(e)}]\n\n"
                    print(f"Warning: Error processing {rel_path} from ZIP: {e}")

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")

    return DocumentConverterResult(
        title="ZIP Archive Contents", text_content=md_content.strip()
    )


def PdfConverter(local_path: str) -> DocumentConverterResult:
    """
    Convert a PDF file to text format.

    Args:
        local_path: Path to PDF file to convert.

    Returns:
        DocumentConverterResult containing extracted text.
    """
    if pdfminer is None:
        return DocumentConverterResult(
            title=None,
            text_content="[Error: pdfminer-six not installed]"
        )

    text_content = pdfminer.high_level.extract_text(local_path)
    return DocumentConverterResult(title=None, text_content=text_content)


def TextConverter(local_path: str) -> DocumentConverterResult:
    """
    Read a text file.

    Args:
        local_path: Path to text file to read.

    Returns:
        DocumentConverterResult containing text content.
    """
    with open(local_path, "r", encoding="utf-8") as f:
        text_content = f.read()
    return DocumentConverterResult(title=None, text_content=text_content)


def JsonConverter(local_path: str) -> DocumentConverterResult:
    """
    Read and format a JSON file.

    Args:
        local_path: Path to JSON file to read.

    Returns:
        DocumentConverterResult containing formatted JSON.
    """
    with open(local_path, "r", encoding="utf-8") as f:
        text_content = json.dumps(
            json.load(f), ensure_ascii=False, indent=2
        )
    return DocumentConverterResult(title=None, text_content=text_content)


def YamlConverter(local_path: str) -> DocumentConverterResult:
    """
    Read a YAML file.

    Args:
        local_path: Path to YAML file to read.

    Returns:
        DocumentConverterResult containing YAML content.
    """
    if yaml is None:
        return DocumentConverterResult(
            title=None,
            text_content="[Error: pyyaml not installed]"
        )

    with open(local_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        text_content = yaml.dump(data, allow_unicode=True, default_flow_style=False)
    return DocumentConverterResult(title=None, text_content=text_content)


def CsvConverter(local_path: str) -> DocumentConverterResult:
    """
    Convert a CSV file to markdown table format.

    Args:
        local_path: Path to CSV file to convert.

    Returns:
        DocumentConverterResult containing markdown table.
    """
    with open(local_path, "r", encoding="utf-8", newline="") as f:
        reader = csv_module.reader(f)
        rows = list(reader)

    if not rows:
        return DocumentConverterResult(title=None, text_content="Empty CSV file")

    md_content = ""
    for i, row in enumerate(rows):
        md_content += "| " + " | ".join(row) + " |\n"
        if i == 0:  # Add a separator after header
            md_content += "|" + "---|" * len(row) + "\n"

    return DocumentConverterResult(title=None, text_content=md_content)


def MarkItDownConverter(local_path: str) -> DocumentConverterResult:
    """
    Convert a file using MarkItDown library (universal converter).

    Args:
        local_path: Path to file to convert.

    Returns:
        DocumentConverterResult containing converted markdown.
    """
    if MarkItDown is None:
        return DocumentConverterResult(
            title=None,
            text_content="[Error: markitdown not installed]"
        )

    md = MarkItDown(enable_plugins=True)
    result = md.convert(local_path)
    return DocumentConverterResult(title=result.title, text_content=result.text_content)
