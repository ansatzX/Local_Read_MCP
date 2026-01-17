# Makefile for local_read_mcp

.PHONY: help install install-dev test format lint clean run

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install production dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  format      - Format code with ruff"
	@echo "  lint        - Lint code with ruff"
	@echo "  check       - Format and lint (pre-commit check)"
	@echo "  clean       - Clean build artifacts"
	@echo "  run         - Run the MCP server (stdio)"
	@echo "  run-http    - Run the MCP server (HTTP)"

# Install production dependencies
install:
	uv pip install -e .

# Install development dependencies
install-dev: install
	uv pip install "pytest>=8.4.1" "pytest-asyncio>=1.0.0"

# Run tests
test:
	uv run pytest src/test/ -v

# Format code
format:
	uv run ruff format src/

# Lint code
lint:
	uv run ruff check src/

# Format and lint (pre-commit check)
check: format lint

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Run the MCP server (stdio transport)
run:
	uv run python -m local_read_mcp.server

# Run the MCP server (HTTP transport)
run-http:
	uv run python -m local_read_mcp.server --transport http --port 8080