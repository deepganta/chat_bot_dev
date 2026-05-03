"""Unit tests for pipelines/rag/Retrieval/llm_utils.py"""
import pytest
from unittest.mock import MagicMock, patch, call
from pipelines.rag.Retrieval.llm_utils import invoke_with_retry

try:
    from openai import RateLimitError, APITimeoutError, APIConnectionError
    _REAL_OPENAI = True
except Exception:
    from pipelines.rag.Retrieval.llm_utils import (
        RateLimitError, APITimeoutError, APIConnectionError
    )
    _REAL_OPENAI = False


# ── Guard: None LLM ──────────────────────────────────────────────────────────

def test_none_llm_raises_value_error():
    with pytest.raises(ValueError, match="None"):
        invoke_with_retry(None, [{"role": "user", "content": "hi"}])


# ── Success path ─────────────────────────────────────────────────────────────

def test_successful_invoke_returns_response():
    mock_response = MagicMock()
    mock_response.content = "Hello from Ford assistant"

    mock_llm = MagicMock()
    mock_llm.bind.return_value = mock_llm
    mock_llm.invoke.return_value = mock_response

    result = invoke_with_retry(mock_llm, [{"role": "user", "content": "hi"}], max_attempts=3)
    assert result.content == "Hello from Ford assistant"


# ── Retry on transient error ─────────────────────────────────────────────────

def test_retries_on_rate_limit_then_succeeds():
    mock_response = MagicMock()
    mock_response.content = "success"

    mock_llm = MagicMock()
    mock_llm.bind.return_value = mock_llm

    # Fail first call, succeed second
    mock_llm.invoke.side_effect = [
        RateLimitError("rate limit", response=MagicMock(), body={}) if _REAL_OPENAI else RateLimitError("rate limit"),
        mock_response,
    ]

    result = invoke_with_retry(mock_llm, [{"role": "user", "content": "hi"}], max_attempts=3)
    assert result.content == "success"
    assert mock_llm.invoke.call_count == 2


# ── Exhausted retries → RuntimeError ─────────────────────────────────────────

def test_raises_runtime_error_after_max_attempts():
    mock_llm = MagicMock()
    mock_llm.bind.return_value = mock_llm

    err = RateLimitError("rate limit", response=MagicMock(), body={}) if _REAL_OPENAI else RateLimitError("rate limit")
    mock_llm.invoke.side_effect = err

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        invoke_with_retry(mock_llm, [{"role": "user", "content": "hi"}], max_attempts=3)


# ── Retry log emitted ────────────────────────────────────────────────────────

def test_retry_warning_logged(caplog):
    import logging
    mock_llm = MagicMock()
    mock_llm.bind.return_value = mock_llm

    mock_response = MagicMock()
    mock_response.content = "ok"

    err = RateLimitError("rate limit", response=MagicMock(), body={}) if _REAL_OPENAI else RateLimitError("rate limit")
    mock_llm.invoke.side_effect = [err, mock_response]

    with caplog.at_level(logging.WARNING, logger="pipelines.rag.Retrieval.llm_utils"):
        invoke_with_retry(mock_llm, [{"role": "user", "content": "hi"}], max_attempts=3)

    assert any("[llm] retry" in r.message for r in caplog.records)
