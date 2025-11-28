"""SQL responder that turns natural language into SQLite queries via GPT-4.

The flow:
1. Use GPT-4 to propose a safe, read-only SQL query against the Ford demo DB.
2. Execute the query on the configured SQLite database.
3. Ask the same GPT-4 model to summarize the result set for the user.
4. Persist the final assistant message to the active conversation.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
import re

try:  # pragma: no cover - optional dependency
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore


DEFAULT_SQL_MODEL = "gpt-4o"
DEFAULT_DB_PATH = Path("data/ford.db")

MessageFactory = Callable[[str, str, str], object]


def _format_history(history: Sequence[object], limit: int = 6) -> str:
    """Format prior conversation messages into a compact string."""

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


def _require_llm() -> None:
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not installed; cannot answer SQL queries.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot access the SQL assistant.")


def _default_llm(model: str) -> ChatOpenAI:
    _require_llm()
    return ChatOpenAI(model=model, temperature=0.0)


def _introspect_schema(conn: sqlite3.Connection) -> str:
    """Return a textual summary of tables/columns for prompt grounding."""
    cursor = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    lines: List[str] = []
    for name, ddl in cursor.fetchall():
        lines.append(ddl.strip().replace("\n", " "))
    return "\n".join(lines)


def _rows_to_display(columns: Sequence[str], rows: Sequence[Sequence[object]], limit: int = 5) -> str:
    """Return a readable text representation of result rows."""

    if not rows:
        return "No rows returned."

    lines = []
    for idx, row in enumerate(rows[:limit], start=1):
        parts = [f"{col}={row[i]}" for i, col in enumerate(columns)]
        lines.append(f"{idx}. " + ", ".join(parts))

    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")

    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """Extract JSON object from LLM response, tolerating code fences."""

    text = raw.strip()
    if not text:
        raise ValueError("Model response was empty.")

    if text.startswith("```"):
        # remove leading ```json or ``` fence
        fence_end = text.find("\n")
        if fence_end != -1:
            text = text[fence_end + 1 :]
        text = text.rstrip("` \n")

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model response was not JSON: {raw}")

    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from model response: {raw}") from exc


def _generate_sql(prompt: str, schema: str, metadata: str, llm: ChatOpenAI) -> Tuple[str, str]:
    system = (
        "You convert user questions into safe, read-only SQLite queries. "
        "Use the provided schema. Available tables and columns:\n"
        "  - plants(id, name, city, state, country, opened_year)\n"
        "  - dealers(id, name, city, state, region)\n"
        "  - vehicles(vin, model, model_year, trim, segment, msrp, plant_id)\n"
        "  - sales(id, vin, dealer_id, sale_date, sale_price, customer_type)\n"
        "Only reference columns exactly as named. Output strict JSON with keys sql and rationale. "
        "Always LIMIT results to at most 50 rows if the query can return many rows. "
        "Do not use DROP/DELETE/UPDATE/INSERT statements."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"Database schema (SQLite DDL):\n{schema}"},
    ]
    if metadata:
        messages.append(
            {"role": "system", "content": f"Reference values from the database:\n{metadata}"}
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "Question:\n"
                f"{prompt}\n\n"
                "Respond with JSON using keys sql and rationale."
            ),
        }
    )
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    data = _parse_json_response(raw)
    sql = data.get("sql", "").strip()
    if not sql:
        raise ValueError("Model did not return SQL.")
    if any(keyword in sql.lower() for keyword in ("drop", "delete", "update", "insert", "alter")):
        raise ValueError(f"Refusing to run potentially destructive SQL: {sql}")
    return sql, data.get("rationale", "")


def _summarize_result(
    prompt: str,
    sql: str,
    rows: Sequence[dict],
    llm: ChatOpenAI,
    note: Optional[str] = None,
) -> str:
    system = (
        "You summarize SQL query results. Provide a concise, factual answer based on the provided rows. "
        "If a note explains that no matching records were found, begin by apologizing and clearly stating "
        "that the requested data was unavailable, then share the closest insight returned. "
        "Do not speculate beyond the data."
    )
    rows_json = json.dumps(rows, indent=2)
    context_note = f"\nNote: {note}" if note else ""
    response = llm.invoke(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"User question:\n{prompt}\n\nSQL executed:\n{sql}{context_note}\n\nRows (JSON):\n{rows_json}"
                ),
            },
        ]
    )
    raw = response.content if hasattr(response, "content") else str(response)
    return raw.strip()


def _build_metadata_summary(conn: sqlite3.Connection) -> str:
    """Gather sample distinct values to ground the LLM in actual data."""

    sections: List[str] = []

    def collect(query: str, label: str) -> None:
        try:
            cur = conn.execute(query)
            values = [row[0] for row in cur.fetchall()]
            if values:
                listed = ", ".join(str(v) for v in values[:10])
                if len(values) > 10:
                    listed += ", ..."
                sections.append(f"{label}: {listed}")
        except Exception:
            pass

    collect("SELECT DISTINCT model FROM vehicles ORDER BY model LIMIT 12", "vehicles.model sample values")
    collect(
        "SELECT DISTINCT segment FROM vehicles ORDER BY segment LIMIT 6",
        "vehicles.segment sample values",
    )
    collect(
        "SELECT DISTINCT customer_type FROM sales ORDER BY customer_type",
        "sales.customer_type values",
    )

    return "\n".join(sections)


def handle_sql_query(
    prompt: str,
    conversation_id: str,
    conversation_store: object,
    message_factory: MessageFactory,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    llm: Optional[ChatOpenAI] = None,
    model: str = DEFAULT_SQL_MODEL,
    history: Optional[Sequence[object]] = None,
) -> Tuple[object, str, Sequence[Sequence[object]]]:
    """Execute the SQL workflow and append the final assistant message.

    Returns the message object, executed SQL string, and the raw rows.

    Parameters
    ----------
    history:
        Optional prior conversation messages (oldest → newest) to maintain context.
    """

    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise ValueError("Prompt must not be empty")

    _require_llm()

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    llm = llm or _default_llm(model=model)

    history_text = _format_history(history or [])
    contextual_prompt = (
        "Conversation context:\n"
        f"{history_text}\n\n"
        f"Current question:\n{clean_prompt}"
        if history_text
        else clean_prompt
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        schema = _introspect_schema(conn)
        metadata = _build_metadata_summary(conn)
        sql, rationale = _generate_sql(contextual_prompt, schema, metadata, llm)

        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description] if cursor.description else []

        no_match_note: Optional[str] = None
        if not rows:
            fallback_sql = (
                "SELECT model, trim, msrp, segment FROM vehicles ORDER BY msrp ASC LIMIT 1"
            )
            cursor = conn.execute(fallback_sql)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description] if cursor.description else []
            sql = fallback_sql
            no_match_note = "Original query returned no rows; fallback query surfaces the closest available record."

        row_dicts = [
            {columns[i]: row[i] for i in range(len(columns))}
            for row in rows
        ]
        summary = _summarize_result(contextual_prompt, sql, row_dicts, llm, note=no_match_note)
        display_rows = _rows_to_display(columns, [tuple(row) for row in rows])

        content_parts = [
            summary,
            "",
            f"SQL: ```sql\n{sql}\n```",
        ]
        if rationale:
            content_parts.insert(1, f"_Model rationale_: {rationale}")
        if display_rows:
            content_parts.append("")
            content_parts.append("Key rows:\n" + display_rows)

        message = message_factory(
            role="assistant",
            content="\n".join(part for part in content_parts if part).strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        append = getattr(conversation_store, "append", None)
        if callable(append):
            append(conversation_id, message)
        else:  # pragma: no cover
            raise AttributeError(
                "conversation_store must expose append(conversation_id, message)"
            )

        return message, sql, rows
    finally:
        conn.close()


__all__ = ["handle_sql_query", "DEFAULT_SQL_MODEL", "DEFAULT_DB_PATH"]
