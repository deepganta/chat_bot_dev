"""Utilities for answering general (non-RAG/SQL) queries with GPT-3.5.

This module centralizes calls to the OpenAI chat model so we can reuse
the same logic across the Streamlit UI and any future API surface.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency at runtime
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

from .llm_utils import invoke_with_retry

DEFAULT_GENERAL_MODEL = "gpt-3.5-turbo"
DEFAULT_TIMEOUT_SEC = 30

SYSTEM_PROMPT = (
    "You are a concise, professional assistant. Answer user questions directly. "
    "If the prompt is unclear or you lack enough context, explicitly mention the "
    "gap instead of inventing facts."
)


# Type alias for constructing `ConversationMessage` without importing handler.py.
MessageFactory = Callable[[str, str, str], object]


def _default_llm(model: str = DEFAULT_GENERAL_MODEL) -> ChatOpenAI:
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed; cannot answer general queries.")
    return ChatOpenAI(model=model, temperature=0.2, timeout=DEFAULT_TIMEOUT_SEC, max_retries=0)


def _history_as_messages(history: Sequence[object], limit: int = 6) -> list[dict[str, str]]:
    """Convert prior turns into role-accurate chat messages."""

    messages: list[dict[str, str]] = []
    for message in history[-limit:]:
        role = str(getattr(message, "role", "assistant")).strip().lower()
        content = str(getattr(message, "content", "") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def handle_general_query(
    prompt: str,
    conversation_id: str,
    conversation_store: object,
    message_factory: Callable[..., object],
    *,
    model: str = DEFAULT_GENERAL_MODEL,
    llm: Optional[ChatOpenAI] = None,
    history: Optional[Sequence[object]] = None,
) -> Tuple[object, str]:
    """Generate a response for a general query.

    Parameters
    ----------
    prompt:
        Raw user prompt that should be answered with general LLM reasoning.
    conversation_id:
        Identifier of the conversation transcript to update.
    conversation_store:
        Store instance that exposes ``append(conversation_id, message)``.
    message_factory:
        Callable that creates a ``ConversationMessage``-like object when given
        ``role``, ``content``, and ``created_at`` keyword arguments.
    model:
        OpenAI chat model identifier. Defaults to GPT-3.5 Turbo.
    llm:
        Optional pre-configured ``ChatOpenAI`` instance.
    history:
        Optional sequence of prior conversation messages (oldest → newest) for
        light conversational memory.

    Returns
    -------
    Tuple[object, str]
        Assistant message object and the raw text returned by the LLM.
    """

    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed; cannot answer general queries.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run general responses.")

    llm = llm or _default_llm(model=model)
    prior_messages = _history_as_messages(history or [])

    response = invoke_with_retry(
        llm,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            *prior_messages,
            {"role": "user", "content": prompt},
        ],
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        max_attempts=3,
    )

    raw_text = response.content if hasattr(response, "content") else str(response)
    content = raw_text.strip() or "I'm not sure how to respond right now."

    timestamp = datetime.now(timezone.utc).isoformat()
    assistant_message = message_factory(
        role="assistant",
        content=content,
        created_at=timestamp,
    )

    return assistant_message, content


__all__ = ["handle_general_query", "DEFAULT_GENERAL_MODEL"]
