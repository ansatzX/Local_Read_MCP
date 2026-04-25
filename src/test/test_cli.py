"""
Unit tests for the CLI entrypoint.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from local_read_mcp import __version__
from local_read_mcp import cli as cli_module


class TestCliMain:
    """Tests for CLI argument handling."""

    def test_main_allows_version_without_input(self, monkeypatch, capsys):
        """--version should work without requiring an input file."""
        monkeypatch.setattr(sys, "argv", ["local-read-mcp", "--version"])

        exit_code = cli_module.main()
        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.out.strip() == f"Local Read MCP v{__version__}"


class TestCliConversion:
    """Tests for CLI conversion behavior."""

    def test_convert_file_uses_processing_pipeline(self, tmp_path):
        """CLI conversion should use the same processing pipeline as MCP tools."""
        input_file = tmp_path / "sample.txt"
        input_file.write_text("hello\nworld\n", encoding="utf-8")

        exit_code = cli_module.convert_file(
            input_path=str(input_file),
            output_dir=str(tmp_path),
            include_page_breaks=False,
            include_metadata=False,
        )

        output_root = tmp_path / ".local_read_mcp"
        output_dirs = list(output_root.iterdir())
        intermediate_path = output_dirs[0] / "intermediate.json"

        with open(intermediate_path, encoding="utf-8") as handle:
            intermediate = json.load(handle)

        first_block = intermediate["blocks"][intermediate["reading_order"][0]]

        assert exit_code == 0
        assert intermediate["source"]["format"] == "text"
        assert first_block["type"] == "text"
        assert first_block["bbox"] == [0, 0, 612, 792]
