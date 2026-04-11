"""
Command Line Interface (CLI) for Local Read MCP.

This module provides a command-line interface for converting documents
using the Local Read MCP processing pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .output_manager import OutputManager
from .index_generator import IndexGenerator
from .markdown_converter import MarkdownConverter
from .intermediate_json import IntermediateJSONBuilder


def convert_file(
    input_path: str,
    output_dir: Optional[str] = None,
    include_page_breaks: bool = True,
    include_metadata: bool = True,
    verbose: bool = False
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
        # Create output directory
        output_manager = OutputManager(base_dir=Path(output_dir) if output_dir else None)
        output_path = output_manager.create_output_dir(input_path)

        if verbose:
            print(f"Output directory: {output_path}")

        # TODO: In a real implementation, we would use the appropriate converter
        # to build the Intermediate JSON. For now, we'll create a placeholder.

        # Create placeholder intermediate JSON
        file_size = input_path_obj.stat().st_size
        builder = IntermediateJSONBuilder(
            source_path=input_path,
            source_format=input_path_obj.suffix[1:] if input_path_obj.suffix else "unknown",
            page_count=1,
            file_size=file_size
        )

        # Try to extract some basic content for demonstration
        # For text files, we'll read the content
        if input_path_obj.suffix.lower() in ['.txt', '.md', '.py', '.sh', '.json', '.csv', '.yaml', '.yml']:
            try:
                with open(input_path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
                builder.add_block(
                    type="paragraph",
                    page=1,
                    bbox=[0, 0, 0, 0],
                    content=content
                )
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not read text content: {e}", file=sys.stderr)

        intermediate = builder.build()

        # Save intermediate JSON
        intermediate_path = output_manager.get_output_path(output_path, "intermediate.json")
        import json
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(intermediate, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"  Saved: {intermediate_path.name}")

        # Generate and save index
        index_generator = IndexGenerator(intermediate)
        index_path = output_manager.get_output_path(output_path, "index.json")
        index_generator.save_to_file(str(index_path))

        if verbose:
            print(f"  Saved: {index_path.name}")

        # Generate and save Markdown
        markdown_converter = MarkdownConverter(
            intermediate,
            include_page_breaks=include_page_breaks,
            include_metadata=include_metadata
        )
        markdown_path = output_manager.get_output_path(output_path, "output.md")
        markdown_converter.save_to_file(str(markdown_path))

        if verbose:
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

    return convert_file(
        input_path=args.input_file,
        output_dir=args.output_dir,
        include_page_breaks=not args.no_page_breaks,
        include_metadata=not args.no_metadata,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())
