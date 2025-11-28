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


DEFAULT_GENERAL_MODEL = "gpt-3.5-turbo"

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
    return ChatOpenAI(model=model, temperature=0.2)


def _format_history(history: Sequence[object], limit: int = 6) -> str:
    """Convert prior conversation turns into a compact context string."""

    if not history:
        return ""

    recent = history[-limit:]
    lines = []
    for message in recent:
        role = getattr(message, "role", "assistant")
        content = getattr(message, "content", "")
        label = "User" if role == "user" else "Assistant"
        content = (content or "").strip()
        if content:
            lines.append(f"{label}: {content}")
    return "\n".join(lines)


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
    """Generate a response for a general query and append it to the conversation.

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
        The persisted assistant message object and the raw text returned by the LLM.
    """

    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed; cannot answer general queries.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run general responses.")

    llm = llm or _default_llm(model=model)

    history_text = _format_history(history or [])
    context_instruction = (
        "Conversation context so far:\n"
        f"{history_text}\n\n"
        "Use this context when replying to the latest user message."
        if history_text
        else ""
    )

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            *(
                [{"role": "system", "content": context_instruction}]
                if context_instruction
                else []
            ),
            {"role": "user", "content": prompt},
        ]
    )

    raw_text = response.content if hasattr(response, "content") else str(response)
    content = raw_text.strip() or "I'm not sure how to respond right now."

    timestamp = datetime.now(timezone.utc).isoformat()
    assistant_message = message_factory(
        role="assistant",
        content=content,
        created_at=timestamp,
    )

    append = getattr(conversation_store, "append", None)
    if callable(append):
        append(conversation_id, assistant_message)
    else:  # pragma: no cover - defensive programming for unexpected store types
        raise AttributeError("conversation_store must provide an append(conversation_id, message) method")

    return assistant_message, content


__all__ = ["handle_general_query", "DEFAULT_GENERAL_MODEL"]
