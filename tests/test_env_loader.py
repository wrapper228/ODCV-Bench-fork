"""Tests for the _load_dotenv inline utility."""
import os
import tempfile
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    """Load .env file into os.environ (only sets vars not already present)."""
    if not env_path.is_file():
        return
    with env_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def test_loads_basic_key(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_LOAD_KEY=hello\n")
    os.environ.pop("TEST_LOAD_KEY", None)
    _load_dotenv(env_file)
    assert os.environ["TEST_LOAD_KEY"] == "hello"
    del os.environ["TEST_LOAD_KEY"]


def test_does_not_override_existing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_NO_OVERRIDE=from_file\n")
    os.environ["TEST_NO_OVERRIDE"] = "from_env"
    _load_dotenv(env_file)
    assert os.environ["TEST_NO_OVERRIDE"] == "from_env"
    del os.environ["TEST_NO_OVERRIDE"]


def test_skips_comments_and_blank_lines(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\n\nTEST_COMMENT_KEY=val\n")
    os.environ.pop("TEST_COMMENT_KEY", None)
    _load_dotenv(env_file)
    assert os.environ["TEST_COMMENT_KEY"] == "val"
    del os.environ["TEST_COMMENT_KEY"]


def test_strips_quotes(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('TEST_QUOTED="quoted_value"\n')
    os.environ.pop("TEST_QUOTED", None)
    _load_dotenv(env_file)
    assert os.environ["TEST_QUOTED"] == "quoted_value"
    del os.environ["TEST_QUOTED"]


def test_missing_file_is_silent(tmp_path):
    _load_dotenv(tmp_path / "nonexistent.env")  # must not raise
