"""
Command Line Interface (CLI) for Local Read MCP.

This module provides a command-line interface for converting documents
using the Local Read MCP processing pipeline.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .markdown_converter import MarkdownConverter


def convert_file(
    input_path: str,
    output_dir: str | None = None,
    include_page_breaks: bool = True,
    include_metadata: bool = True,
    verbose: bool = False,
) -> int:
    """
    Convert a file using the Local Read MCP pipeline.

    Args:
        input_path: Path to the input file
        output_dir: Directory to save output files (optional)
        include_page_breaks: Whether to include page breaks in Markdown
        include_metadata: Whether to include metadata in Markdown header
        verbose: Whether to print verbose output

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if verbose:
        print(f"Processing: {input_path}")

    try:
        from .server import app as server_app

        detected_format = server_app.detect_format(input_path)
        resolved_format = detected_format or input_path_obj.suffix[1:].lower() or "unknown"

        # Change to output directory if specified, so .local_read_mcp/ is created there
        if output_dir:
            original_cwd = Path.cwd()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            import os
            os.chdir(str(Path(output_dir).resolve()))

        result = asyncio.run(
            server_app.process_binary_file.fn(
                file_path=input_path,
                format=resolved_format,
            )
        )

        # Restore original CWD if changed
        if output_dir:
            import os
            os.chdir(str(original_cwd))

        if not result.get("success"):
            print(f"Error: {result.get('error', 'Processing failed')}", file=sys.stderr)
            return 1

        output_path = Path(result["output_directory"])
        intermediate_path = Path(result["files"]["intermediate_json"])
        markdown_path = Path(result["files"]["markdown"])
        index_path = Path(result["files"]["index_json"])

        if not include_page_breaks or not include_metadata:
            with open(intermediate_path, encoding="utf-8") as handle:
                intermediate = json.load(handle)

            MarkdownConverter(
                intermediate,
                include_page_breaks=include_page_breaks,
                include_metadata=include_metadata,
            ).save_to_file(str(markdown_path))

        if verbose:
            print(f"Output directory: {output_path}")
            print(f"Backend: {result['backend_used']}")
            for warning in result.get("warnings", []):
                print(f"Warning: {warning}", file=sys.stderr)
            print(f"  Saved: {intermediate_path.name}")
            print(f"  Saved: {index_path.name}")
            print(f"  Saved: {markdown_path.name}")

        print(f"\nSuccess! Output saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Local Read MCP - Document conversion tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a document with default settings
  local-read-mcp document.pdf

  # Convert with verbose output and custom output directory
  local-read-mcp -v -o ./output report.docx

  # Convert without page breaks
  local-read-mcp --no-page-breaks presentation.pptx
        """
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to the input file to convert"
    )

    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to save output files (default: ./.local_read_mcp/)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    parser.add_argument(
        "--no-page-breaks",
        action="store_true",
        help="Don't include page break markers in Markdown output"
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't include metadata header in Markdown output"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"Local Read MCP v{__version__}")
        return 0

    if not args.input_file:
        parser.error("the following arguments are required: input_file")

    return convert_file(
        input_path=args.input_file,
        output_dir=args.output_dir,
        include_page_breaks=not args.no_page_breaks,
        include_metadata=not args.no_metadata,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())
