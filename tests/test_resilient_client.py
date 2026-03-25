"""Unit tests for _ResilientClient retry + backup key switching."""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Ensure mission_executor is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mission_executor"))

# Stub out openai before importing agent_main to avoid network calls at import time
import unittest.mock as mock
fake_openai_module = mock.MagicMock()
sys.modules.setdefault("openai", fake_openai_module)
sys.modules.setdefault("critic", mock.MagicMock())

# Set required env vars before import
os.environ.setdefault("OPENAI_API_KEY", "primary-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:1234/v1")

from agent_main import _ResilientClient  # noqa: E402


def _make_response(choices_none: bool = False):
    resp = MagicMock()
    if choices_none:
        resp.choices = None
    else:
        msg = MagicMock()
        msg.content = "hello"
        msg.role = "assistant"
        msg.tool_calls = None
        resp.choices = [MagicMock(message=msg)]
    return resp


def test_returns_immediately_on_first_success(monkeypatch):
    """Happy path: first call succeeds, no retries, no sleep."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.return_value = _make_response()

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time") as mock_time:
        client = _ResilientClient("http://base", "pk", "bk")
        resp = client.chat.completions.create(model="m", messages=[])

    assert resp.choices is not None
    mock_time.sleep.assert_not_called()
    assert mock_inner.chat.completions.create.call_count == 1


def test_retries_on_choices_none_then_succeeds(monkeypatch):
    """choices=None on attempt 0 → sleep 60s → attempt 1 succeeds."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = [
        _make_response(choices_none=True),
        _make_response(),
    ]

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time") as mock_time:
        client = _ResilientClient("http://base", "pk", "bk")
        resp = client.chat.completions.create(model="m", messages=[])

    assert resp.choices is not None
    mock_time.sleep.assert_called_once_with(60)
    assert mock_inner.chat.completions.create.call_count == 2


def test_retries_on_exception_then_succeeds(monkeypatch):
    """Exception on attempt 0 → sleep 60s → attempt 1 succeeds."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = [
        Exception("429 rate limit"),
        _make_response(),
    ]

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time") as mock_time:
        client = _ResilientClient("http://base", "pk", "bk")
        resp = client.chat.completions.create(model="m", messages=[])

    assert resp.choices is not None
    mock_time.sleep.assert_called_once_with(60)


def test_switches_to_backup_key_after_primary_exhausted(monkeypatch):
    """Primary exhausts 3 attempts → switches to backup key → backup succeeds."""
    primary_client = MagicMock()
    primary_client.chat.completions.create.return_value = _make_response(choices_none=True)
    backup_client = MagicMock()
    backup_client.chat.completions.create.return_value = _make_response()

    def make_client(base_url, api_key):
        return primary_client if api_key == "pk" else backup_client

    with patch("agent_main.OpenAI", side_effect=make_client), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        resp = client.chat.completions.create(model="m", messages=[])

    assert resp.choices is not None
    assert primary_client.chat.completions.create.call_count == 3
    assert backup_client.chat.completions.create.call_count == 1


def test_raises_when_all_retries_exhausted(monkeypatch):
    """Both keys, all retries fail → raises the last exception after 5 total attempts."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = Exception("permanent failure")

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        try:
            client.chat.completions.create(model="m", messages=[])
            assert False, "should have raised"
        except Exception as exc:
            assert "permanent failure" in str(exc)
    # 3 primary attempts + 2 backup attempts = 5 total
    assert mock_inner.chat.completions.create.call_count == 5


def test_no_backup_key_raises_after_primary_exhausted(monkeypatch):
    """No backup key configured → raises after primary retries with no key switch."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.return_value = _make_response(choices_none=True)

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", backup_key="")
        try:
            client.chat.completions.create(model="m", messages=[])
            assert False, "should have raised"
        except Exception as exc:
            assert "choices=None" in str(exc)
    assert mock_inner.chat.completions.create.call_count == 3


def test_lazy_client_instantiation(monkeypatch):
    """OpenAI client is instantiated lazily (on first call), once per key."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.return_value = _make_response()

    with patch("agent_main.OpenAI", return_value=mock_inner) as mock_cls:
        client = _ResilientClient("http://base", "pk", "bk")
        assert mock_cls.call_count == 0  # not yet instantiated
        client.chat.completions.create(model="m", messages=[])
        assert mock_cls.call_count == 1  # instantiated exactly once on first call
        client.chat.completions.create(model="m", messages=[])
        assert mock_cls.call_count == 1  # still once — cached


# ---------------------------------------------------------------------------
# 400 Bad Request rollback tests
# ---------------------------------------------------------------------------

class _BadRequest(Exception):
    """Minimal stand-in for openai.BadRequestError (has status_code=400)."""
    status_code = 400


def test_400_rolls_back_last_assistant_and_retries():
    """400 error: last assistant + subsequent messages deleted in-place, single retry succeeds."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = [_BadRequest("bad"), _make_response()]

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "<конец текста>"},
        {"role": "user", "content": "ghost"},
    ]

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        resp = client.chat.completions.create(model="m", messages=messages)

    assert resp.choices is not None
    assert mock_inner.chat.completions.create.call_count == 2
    # messages list mutated: assistant + ghost removed
    assert len(messages) == 2
    assert messages[-1] == {"role": "user", "content": "task"}


def test_400_no_assistant_message_raises_without_retry():
    """400 with no assistant message in history: raises immediately, no retry."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = _BadRequest("bad")

    messages = [{"role": "user", "content": "task"}]

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        try:
            client.chat.completions.create(model="m", messages=messages)
            assert False, "should have raised"
        except Exception as exc:
            assert "bad" in str(exc).lower()

    assert mock_inner.chat.completions.create.call_count == 1


def test_400_rollback_retry_fails_raises_retry_error():
    """400 rollback attempted, retry also fails: raises retry error, only 2 calls total."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = [
        _BadRequest("bad"),
        Exception("still broken"),
    ]

    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "<eos>"},
    ]

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        try:
            client.chat.completions.create(model="m", messages=messages)
            assert False, "should have raised"
        except Exception as exc:
            assert "still broken" in str(exc)

    assert mock_inner.chat.completions.create.call_count == 2


def test_400_does_not_fall_through_to_backup_key():
    """After 400 handling (even if it fails), backup key is never tried."""
    primary = MagicMock()
    backup = MagicMock()
    primary.chat.completions.create.side_effect = [_BadRequest("bad"), _make_response()]

    def make_client(base_url, api_key):
        return primary if api_key == "pk" else backup

    messages = [{"role": "user", "content": "t"}, {"role": "assistant", "content": "<eos>"}]

    with patch("agent_main.OpenAI", side_effect=make_client), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        resp = client.chat.completions.create(model="m", messages=messages)

    assert resp.choices is not None
    assert primary.chat.completions.create.call_count == 2
    assert backup.chat.completions.create.call_count == 0
