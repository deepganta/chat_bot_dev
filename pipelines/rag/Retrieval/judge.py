"""Judge module to classify user prompts for routing.

The judge decides whether a prompt should be answered via:
  - general LLM reasoning ("general")
  - RAG retrieval from the vector store ("rag")
  - SQL query generation + execution ("sql")

It uses GPT-3.5 by default, but falls back to lightweight keyword
heuristics when the OpenAI API key or dependency is unavailable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

try:  # pragma: no cover - optional dependency at runtime
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - keep import optional during tests
    ChatOpenAI = None  # type: ignore


DEFAULT_MODEL = "gpt-3.5-turbo"


class Decision(str, Enum):
    GENERAL = "general"
    RAG = "rag"
    SQL = "sql"


class VerdictSchema(BaseModel):
    """Structured response contract for the judge LLM."""

    decision: Decision = Field(..., description="Routing decision")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., description="Short explanation")


@dataclass
class JudgeVerdict:
    """Normalized verdict returned by :func:`judge_prompt`."""

    decision: Decision
    confidence: float
    rationale: str
    raw_response: Optional[str] = None


SYSTEM_PROMPT = """You are a routing assistant for a hybrid retrieval system.
Classify the user inquiry into exactly one of three options:
  - "general": a plain question that can be answered with general knowledge.
  - "rag": the user is asking about domain documents stored in a vector DB.
  - "sql": the user needs structured business data that lives in a SQL database.

Return strict JSON with keys decision, confidence, rationale. Confidence must
be a number between 0 and 1 reflecting your certainty.
"""


def _default_llm(model: str = DEFAULT_MODEL):
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not available")
    return ChatOpenAI(model=model, temperature=0.0)


def judge_prompt(
    prompt: str,
    model: str = DEFAULT_MODEL,
    llm: Optional[ChatOpenAI] = None,
) -> JudgeVerdict:
    """Classify the prompt using GPT-3.5 or heuristics as a fallback."""

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must not be empty")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or ChatOpenAI is None:
        return _heuristic_judgement(prompt)

    llm = llm or _default_llm(model)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    raw_content: str = response.content if hasattr(response, "content") else str(response)

    try:
        data = json.loads(raw_content)
        parsed = VerdictSchema.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        # Fallback: attempt simple heuristic if parsing fails
        fallback = _heuristic_judgement(prompt)
        fallback.raw_response = raw_content
        return fallback

    return JudgeVerdict(
        decision=parsed.decision,
        confidence=parsed.confidence,
        rationale=parsed.rationale,
        raw_response=raw_content,
    )


KEYWORDS_SQL = {"sql", "database", "table", "select", "query", "queries", "schema"}
KEYWORDS_RAG = {"document", "docs", "policy", "report", "manual", "vector", "embed"}


def _heuristic_judgement(prompt: str) -> JudgeVerdict:
    lower = prompt.lower()

    if any(keyword in lower for keyword in KEYWORDS_SQL):
        decision = Decision.SQL
        rationale = "Detected SQL-oriented keywords."
    elif any(keyword in lower for keyword in KEYWORDS_RAG):
        decision = Decision.RAG
        rationale = "Detected document/vector store related keywords."
    else:
        decision = Decision.GENERAL
        rationale = "Defaulting to general reasoning path."

    return JudgeVerdict(decision=decision, confidence=0.4, rationale=rationale)


__all__ = [
    "Decision",
    "JudgeVerdict",
    "judge_prompt",
]
