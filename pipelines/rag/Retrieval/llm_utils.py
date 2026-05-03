"""Shared LLM invocation helpers with retry and timeout handling."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:  # pragma: no cover - optional dependency in some test environments
    from openai import APIConnectionError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover
    class RateLimitError(Exception):  # type: ignore[no-redef]
        """Fallback class when openai package is unavailable."""

    class APITimeoutError(Exception):  # type: ignore[no-redef]
        """Fallback class when openai package is unavailable."""

    class APIConnectionError(Exception):  # type: ignore[no-redef]
        """Fallback class when openai package is unavailable."""


RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError)
log = logging.getLogger(__name__)


def _log_retry(retry_state) -> None:
    err = retry_state.outcome.exception() if retry_state.outcome else None
    log.warning(
        "[llm] retry attempt=%d reason=%s",
        retry_state.attempt_number,
        err,
    )


def invoke_with_retry(
    llm: Any,
    messages: Sequence[dict[str, str]],
    *,
    timeout_sec: int = 30,
    max_attempts: int = 3,
):
    """Invoke an LLM with retry/timeout behavior for transient OpenAI failures."""

    if llm is None:
        raise ValueError("LLM instance must not be None")

    caller = llm.bind(timeout=timeout_sec) if hasattr(llm, "bind") else llm

    @retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=_log_retry,
    )
    def _invoke():
        return caller.invoke(messages)

    try:
        return _invoke()
    except RETRYABLE_EXCEPTIONS as exc:
        raise RuntimeError(f"LLM unavailable after {max_attempts} attempts") from exc


__all__ = ["invoke_with_retry"]
