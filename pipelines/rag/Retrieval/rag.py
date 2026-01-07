"""Utilities for answering RAG-classified queries with the existing vector store.

This module reuses ingestion helpers to load the configured Chroma index and
route prompts through a RetrievalQA chain backed by GPT-3.5 (default). The
resulting answer is persisted to the conversation transcript alongside a
summary of the sources that supported the response.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

try:  # pragma: no cover
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

from pipelines.rag.ingestion.qa import build_vectorstore, load_cfg


DEFAULT_RAG_MODEL = "gpt-3.5-turbo"
DEFAULT_CONFIG_PATH = Path("Configs/corpus.yaml")

# Type alias mirroring the general responder without importing handler directly.
MessageFactory = Callable[[str, str, str], object]


def _default_llm(model: str = DEFAULT_RAG_MODEL) -> ChatOpenAI:
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed; cannot answer RAG queries.")
    return ChatOpenAI(model=model, temperature=0.0)


def _format_sources(sources: Sequence[object]) -> str:
    if not sources:
        return "Sources: (none returned)"

    lines = ["Sources:"]
    for idx, doc in enumerate(sources, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        url = metadata.get("url") or metadata.get("source") or "<no-source>"
        chunk_id = metadata.get("chunk_id") or metadata.get("order")
        if chunk_id:
            lines.append(f"[{idx}] {url} (chunk={chunk_id})")
        else:
            lines.append(f"[{idx}] {url}")
    return "\n".join(lines)


def _format_history(history: Sequence[object], limit: int = 6) -> str:
    """Turn prior conversation turns into a lightweight context string."""

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


def handle_rag_query(
    prompt: str,
    conversation_id: str,
    conversation_store: object,
    message_factory: MessageFactory,
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    retriever: Optional[object] = None,
    llm: Optional[ChatOpenAI] = None,
    chain: Optional[object] = None,
    top_k: Optional[int] = None,
    model: str = DEFAULT_RAG_MODEL,
    history: Optional[Sequence[object]] = None,
) -> Tuple[object, str, Sequence[object]]:
    """Generate a retrieval-augmented response and append it to the conversation.

    Returns a tuple so the caller can log the raw pieces if needed.

    Parameters
    ----------
    history:
        Optional prior conversation messages (oldest → newest) to maintain context.
    """

    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed; cannot answer RAG queries.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot execute RAG pipeline.")

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"RAG config not found: {cfg_path}")

    cfg = load_cfg(str(cfg_path))
    k = top_k or cfg.get("retriever_k", 4)

    vectorstore = getattr(conversation_store, "_vectorstore_cache", None)
    if vectorstore is None:
        vectorstore = build_vectorstore(cfg)
        setattr(conversation_store, "_vectorstore_cache", vectorstore)

    if retriever is None:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    llm = llm or _default_llm(model=model)

    history_text = _format_history(history or [])
    query_text = (
        "Conversation context:\n"
        f"{history_text}\n\n"
        f"Current question:\n{prompt}"
        if history_text
        else prompt
    )

    if chain is not None:  # pragma: no cover - allow injected chains for tests
        raw_result = chain.invoke({"query": query_text})
        answer = (raw_result.get("result") or "").strip()
        sources: Sequence[object] = raw_result.get("source_documents") or []
    else:
        sources = retriever.get_relevant_documents(query_text)
        context = "\n\n".join(
            (getattr(doc, "page_content", "") or "").strip()
            for doc in sources
            if getattr(doc, "page_content", None)
        ).strip()

        system_prompt = (
            "You are a Ford company assistant. Answer using ONLY the provided context snippets. "
            "If the context does not contain the answer, say \"I couldn't find this in the indexed "
            "Ford documents.\" Do not speculate. Be concise and helpful."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{query_text}"
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        answer = (response.content if hasattr(response, "content") else str(response)).strip()

    if not answer:
        answer = "I could not find a definitive answer in the indexed documents."

    timestamp = datetime.now(timezone.utc).isoformat()
    content = answer + "\n\n" + _format_sources(sources)
    assistant_message = message_factory(
        role="assistant",
        content=content.strip(),
        created_at=timestamp,
    )

    append = getattr(conversation_store, "append", None)
    if callable(append):
        append(conversation_id, assistant_message)
    else:  # pragma: no cover - defensive guard
        raise AttributeError("conversation_store must provide an append(conversation_id, message) method")

    return assistant_message, answer, sources


__all__ = ["handle_rag_query", "DEFAULT_RAG_MODEL"]
