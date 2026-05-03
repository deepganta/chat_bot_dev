"""Basic backend utilities for the retrieval UI.

At this stage we simply persist incoming user queries and echo
them back. The persistence format is JSONL so it stays compatible
with future analytics or replay tooling.

To generate a conversation encryption key:
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import uuid

try:  # pragma: no cover - optional dependency at runtime
    from cryptography.fernet import Fernet, InvalidToken
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore[assignment]
    InvalidToken = ValueError  # type: ignore[assignment]

from .enrichment import fetch_dealer_offers
from .guardrails import sanitize_input
from .judge import Decision, FORD_MODELS, JudgeVerdict, judge_prompt
from .pii import redact_pii
from .response_formatter import format_response


DEFAULT_LOG_PATH = Path("data/queries/query_log.jsonl")
DEFAULT_CONVERSATION_DIR = Path("data/conversations")
log = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """Container for one captured user query."""

    conversation_id: str
    prompt: str
    created_at: str


@dataclass
class ConversationMessage:
    """One turn in a conversation transcript."""

    role: str
    content: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class QueryStore:
    """Append-only JSONL store for user queries."""

    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: QueryRecord) -> None:
        """Persist a record to disk."""
        with self._path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


class ConversationStore:
    """Persist conversation transcripts as JSONL files."""

    def __init__(self, base_dir: Path = DEFAULT_CONVERSATION_DIR) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._fernet = self._build_fernet()

    def create(self, conversation_id: Optional[str] = None) -> str:
        resolved_conversation_id = conversation_id or uuid.uuid4().hex
        path = self._path(resolved_conversation_id)
        path.touch(exist_ok=True)
        return resolved_conversation_id

    def append(self, conversation_id: str, message: ConversationMessage) -> None:
        path = self._path(conversation_id)
        payload = json.dumps(message.to_dict(), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as sink:
            if self._fernet is None:
                log.warning("CONVERSATION_KEY not set; conversation data is stored unencrypted")
                sink.write(payload + "\n")
            else:
                token = self._fernet.encrypt(payload.encode("utf-8")).decode("utf-8")
                sink.write(token + "\n")

    def load(self, conversation_id: str) -> Iterable[ConversationMessage]:
        path = self._path(conversation_id)
        if not path.exists():
            return []
        messages = []
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                if self._fernet is None:
                    payload_text = line
                else:
                    try:
                        payload_text = self._fernet.decrypt(line.encode("utf-8")).decode("utf-8")
                    except InvalidToken as exc:
                        raise ValueError(f"Failed to decrypt conversation data in {path}") from exc
                payload = json.loads(payload_text)
                messages.append(ConversationMessage(**payload))
        return messages

    def path_for(self, conversation_id: str) -> Path:
        return self._path(conversation_id)

    def _path(self, conversation_id: str) -> Path:
        return self._base_dir / f"{conversation_id}.jsonl"

    def _build_fernet(self):
        key = os.getenv("CONVERSATION_KEY")
        if not key:
            return None
        if Fernet is None:
            raise RuntimeError(
                "cryptography is required when CONVERSATION_KEY is set."
            )
        try:
            return Fernet(key)
        except Exception as exc:
            raise RuntimeError(
                "Invalid CONVERSATION_KEY; expected a base64 Fernet key."
            ) from exc


@dataclass
class HandlerResult:
    """Pack the outcome of handling a user prompt."""

    conversation_id: str
    user_message: ConversationMessage
    assistant_message: ConversationMessage
    user_zip: Optional[str] = None
    verdict: Optional[JudgeVerdict] = None


SUGGESTIONS_BY_DECISION: dict[Decision, list[str]] = {
    Decision.RAG: [
        "What is Ford's warranty policy?",
        "Tell me about Ford's EV lineup",
        "What support options does Ford offer?",
    ],
    Decision.SQL: [
        "Show me the most affordable Ford models",
        "Which dealers are in my area?",
        "Compare F-150 trims by price",
    ],
    Decision.GENERAL: [
        "What Ford vehicles are available?",
        "How do I schedule a test drive?",
        "What financing options does Ford offer?",
    ],
}


def _detect_model(prompt: str) -> Optional[str]:
    lower = (prompt or "").lower()
    for model in FORD_MODELS:
        if model in lower:
            return model
    return None


def _get_suggestions(decision: Optional[Decision]) -> list[str]:
    if decision is None:
        return SUGGESTIONS_BY_DECISION[Decision.GENERAL]
    return list(SUGGESTIONS_BY_DECISION.get(decision, SUGGESTIONS_BY_DECISION[Decision.GENERAL]))


def start_new_conversation(store: Optional[ConversationStore] = None) -> str:
    """Create a fresh conversation transcript and return its ID."""

    target_store = store or ConversationStore()
    conversation_id = target_store.create()
    return conversation_id


def handle_user_query(
    prompt: str,
    conversation_id: Optional[str] = None,
    user_zip: Optional[str] = None,
    store: Optional[QueryStore] = None,
    conversation_store: Optional[ConversationStore] = None,
    enable_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    llm: Optional[object] = None,
) -> HandlerResult:
    """Persist the prompt and return the stored records.

    Parameters
    ----------
    prompt:
        Raw text supplied by the UI.
    store:
        Optional custom store implementation (handy for testing).
    """
    session_id = conversation_id or uuid.uuid4().hex
    clean_prompt = sanitize_input(prompt, session_id=session_id)

    target_store = store or QueryStore()
    conversations = conversation_store or ConversationStore()

    if not conversation_id:
        conversation_id = conversations.create(conversation_id=session_id)

    redacted_prompt, was_redacted, detected_pii_types = redact_pii(clean_prompt)
    if was_redacted:
        detected_label = "[" + ", ".join(detected_pii_types) + "]"
        log.warning(
            "[pii] WARNING detected=%s conversation=%s",
            detected_label,
            conversation_id,
        )

    now = datetime.now(timezone.utc).isoformat()

    record = QueryRecord(
        conversation_id=conversation_id,
        prompt=redacted_prompt,
        created_at=now,
    )
    target_store.append(record)

    user_message = ConversationMessage(role="user", content=record.prompt, created_at=now)
    conversations.append(conversation_id, user_message)

    conversation_history = list(conversations.load(conversation_id))
    context_history = conversation_history[:-1] if conversation_history else []

    verdict: Optional[JudgeVerdict] = None
    if enable_judge:
        verdict = judge_prompt(prompt=record.prompt, model=judge_model, llm=llm)
        if verdict.confidence < 0.60 and verdict.decision in {Decision.GENERAL, Decision.SQL}:
            log.info(
                "[judge] low-confidence %s overridden to RAG confidence=%.2f",
                verdict.decision.value.upper(),
                verdict.confidence,
            )
            verdict = JudgeVerdict(
                decision=Decision.RAG,
                confidence=verdict.confidence,
                rationale=(
                    f"{verdict.rationale} "
                    "[override applied: low-confidence non-RAG routing redirected to RAG]"
                ),
                raw_response=verdict.raw_response,
            )

    assistant_reply: ConversationMessage
    raw_answer: Optional[str] = None
    assistant_draft: Optional[ConversationMessage] = None
    general_raw_response: Optional[str] = None
    rag_sources: Optional[list] = None
    rag_answer: Optional[str] = None
    sql_query: Optional[str] = None
    sql_row_count: Optional[int] = None

    if verdict and verdict.decision == Decision.GENERAL:
        try:
            from .general import handle_general_query

            assistant_draft, general_raw_response = handle_general_query(
                prompt=record.prompt,
                conversation_id=conversation_id,
                conversation_store=conversations,
                message_factory=ConversationMessage,
                history=context_history,
            )
            raw_answer = general_raw_response
        except Exception as exc:
            assistant_reply = ConversationMessage(
                role="assistant",
                content=(
                    "We captured your question, but the general responder is unavailable right now.\n"
                    f"Reason: {exc}"
                ),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            conversations.append(conversation_id, assistant_reply)
    elif verdict and verdict.decision == Decision.RAG:
        try:
            from .rag import handle_rag_query

            assistant_draft, rag_answer, rag_sources = handle_rag_query(
                prompt=record.prompt,
                conversation_id=conversation_id,
                conversation_store=conversations,
                message_factory=ConversationMessage,
                history=context_history,
            )
            raw_answer = rag_answer
        except Exception as exc:
            assistant_reply = ConversationMessage(
                role="assistant",
                content=(
                    "We captured your question, but the retrieval responder is unavailable right now.\n"
                    f"Reason: {exc}"
                ),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            conversations.append(conversation_id, assistant_reply)
    elif verdict and verdict.decision == Decision.SQL:
        try:
            from .sql import handle_sql_query

            sql_message, sql_query, sql_rows = handle_sql_query(
                prompt=record.prompt,
                conversation_id=conversation_id,
                conversation_store=conversations,
                message_factory=ConversationMessage,
                history=context_history,
            )
            assistant_draft = sql_message
            raw_answer = sql_message.content
            sql_row_count = len(sql_rows) if sql_rows is not None else 0
        except Exception as exc:
            assistant_reply = ConversationMessage(
                role="assistant",
                content=(
                    "We captured your question, but the SQL responder is unavailable right now.\n"
                    f"Reason: {exc}"
                ),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            conversations.append(conversation_id, assistant_reply)
    else:
        assistant_reply_content = (
            "Thanks! Your question has been captured. A response workflow will pick it up shortly."
            if verdict is None
            else (
                "Routing decision: {path}. (Confidence {confidence:.2f})\n"
                "The specialized agent will respond soon."
            ).format(path=verdict.decision.value, confidence=verdict.confidence)
        )

        assistant_reply = ConversationMessage(
            role="assistant",
            content=assistant_reply_content,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        conversations.append(conversation_id, assistant_reply)

    if verdict is not None and raw_answer is not None:
        detected_model = _detect_model(record.prompt)
        dealer_offers = []
        if detected_model and user_zip:
            dealer_offers = fetch_dealer_offers(
                zip_code=user_zip,
                model=detected_model,
            )
        suggestions = _get_suggestions(verdict.decision)
        final_content = format_response(raw_answer, dealer_offers, suggestions)

        assistant_reply = ConversationMessage(
            role="assistant",
            content=final_content,
            created_at=assistant_draft.created_at if assistant_draft else datetime.now(timezone.utc).isoformat(),
        )
        conversations.append(conversation_id, assistant_reply)
    elif assistant_reply is None:
        assistant_reply = ConversationMessage(
            role="assistant",
            content="We captured your question, but no response path produced an answer.",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        conversations.append(conversation_id, assistant_reply)

    log.info(
        "[retrieval] conversation=%s captured prompt @ %s: %s",
        conversation_id,
        record.created_at,
        record.prompt,
    )
    if verdict is not None:
        log.info(
            "[retrieval] verdict -> decision=%s confidence=%.2f",
            verdict.decision.value,
            verdict.confidence,
        )
        if verdict.raw_response:
            log.debug("[retrieval] raw judge response: %s", verdict.raw_response)
        if general_raw_response:
            log.debug("[retrieval] general-response raw content: %s", general_raw_response)
        if rag_answer is not None:
            log.debug("[retrieval] rag-response answer: %s", rag_answer)
            if rag_sources:
                summaries = []
                for doc in rag_sources:
                    metadata = getattr(doc, "metadata", {}) or {}
                    url = metadata.get("url") or metadata.get("source") or "<no-source>"
                    summaries.append(url)
                log.debug("[retrieval] rag-response sources: %s", ", ".join(summaries))
        if sql_query is not None:
            log.debug("[retrieval] sql-response query: %s", sql_query)
            if sql_row_count is not None:
                log.debug("[retrieval] sql-response rows: %d", sql_row_count)

    return HandlerResult(
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_reply,
        user_zip=user_zip,
        verdict=verdict,
    )


__all__ = [
    "handle_user_query",
    "start_new_conversation",
    "QueryRecord",
    "QueryStore",
    "ConversationMessage",
    "ConversationStore",
    "HandlerResult",
    "JudgeVerdict",
]
