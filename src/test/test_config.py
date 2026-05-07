"""Tests for Local Read MCP configuration loading."""

from pathlib import Path

from local_read_mcp.config import Config


def test_config_loads_dotenv_from_explicit_project_root(monkeypatch, tmp_path):
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VISION_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "# Local vision config",
                "VISION_API_KEY=from-dotenv",
                "VISION_BASE_URL='https://example.test/v1'",
                'VISION_MODEL="vision-model"',
            ]
        ),
        encoding="utf-8",
    )

    config = Config(dotenv_path=tmp_path)

    assert config.api_key == "from-dotenv"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "vision-model"
    assert config.vision_enabled is True


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_API_KEY", "from-env")
    (tmp_path / ".env").write_text("VISION_API_KEY=from-dotenv\n", encoding="utf-8")

    config = Config(dotenv_path=tmp_path)

    assert config.api_key == "from-env"
