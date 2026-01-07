"""Basic backend utilities for the retrieval UI.

At this stage we simply persist incoming user queries and echo
them back. The persistence format is JSONL so it stays compatible
with future analytics or replay tooling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import uuid

from .judge import Decision, JudgeVerdict, judge_prompt


DEFAULT_LOG_PATH = Path("data/queries/query_log.jsonl")
DEFAULT_CONVERSATION_DIR = Path("data/conversations")


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

    def create(self) -> str:
        conversation_id = uuid.uuid4().hex
        path = self._path(conversation_id)
        path.touch(exist_ok=True)
        return conversation_id

    def append(self, conversation_id: str, message: ConversationMessage) -> None:
        path = self._path(conversation_id)
        with path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

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
                payload = json.loads(line)
                messages.append(ConversationMessage(**payload))
        return messages

    def path_for(self, conversation_id: str) -> Path:
        return self._path(conversation_id)

    def _path(self, conversation_id: str) -> Path:
        return self._base_dir / f"{conversation_id}.jsonl"


@dataclass
class HandlerResult:
    """Pack the outcome of handling a user prompt."""

    conversation_id: str
    user_message: ConversationMessage
    assistant_message: ConversationMessage
    verdict: Optional[JudgeVerdict] = None


def start_new_conversation(store: Optional[ConversationStore] = None) -> str:
    """Create a fresh conversation transcript and return its ID."""

    target_store = store or ConversationStore()
    conversation_id = target_store.create()
    return conversation_id


def handle_user_query(
    prompt: str,
    conversation_id: Optional[str] = None,
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

    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    target_store = store or QueryStore()
    conversations = conversation_store or ConversationStore()

    if not conversation_id:
        conversation_id = conversations.create()

    now = datetime.now(timezone.utc).isoformat()

    record = QueryRecord(
        conversation_id=conversation_id,
        prompt=prompt.strip(),
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

    assistant_reply: ConversationMessage
    general_raw_response: Optional[str] = None
    rag_sources: Optional[list] = None
    rag_answer: Optional[str] = None
    sql_query: Optional[str] = None
    sql_row_count: Optional[int] = None

    if verdict and verdict.decision == Decision.GENERAL:
        try:
            from .general import handle_general_query

            assistant_reply, general_raw_response = handle_general_query(
                prompt=record.prompt,
                conversation_id=conversation_id,
                conversation_store=conversations,
                message_factory=ConversationMessage,
                history=context_history,
            )
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

            assistant_reply, rag_answer, rag_sources = handle_rag_query(
                prompt=record.prompt,
                conversation_id=conversation_id,
                conversation_store=conversations,
                message_factory=ConversationMessage,
                history=context_history,
            )
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
            assistant_reply = sql_message
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

    # Initial milestone: print the conversation id and prompt for operator visibility.
    print(
        f"[retrieval] conversation={conversation_id} captured prompt @ {record.created_at}: {record.prompt}"
    )
    if verdict is not None:
        print(
            f"[retrieval] verdict → decision={verdict.decision.value} confidence={verdict.confidence:.2f}"
        )
        if verdict.raw_response:
            print(f"[retrieval] raw judge response: {verdict.raw_response}")
        if general_raw_response:
            print(f"[retrieval] general-response raw content: {general_raw_response}")
        if rag_answer is not None:
            print(f"[retrieval] rag-response answer: {rag_answer}")
            if rag_sources:
                summaries = []
                for doc in rag_sources:
                    metadata = getattr(doc, "metadata", {}) or {}
                    url = metadata.get("url") or metadata.get("source") or "<no-source>"
                    summaries.append(url)
                print(f"[retrieval] rag-response sources: {', '.join(summaries)}")
        if sql_query is not None:
            print(f"[retrieval] sql-response query: {sql_query}")
            if sql_row_count is not None:
                print(f"[retrieval] sql-response rows: {sql_row_count}")

    return HandlerResult(
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_reply,
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
