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
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from .llm_utils import invoke_with_retry

try:  # pragma: no cover - optional dependency at runtime
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - keep import optional during tests
    ChatOpenAI = None  # type: ignore


DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TIMEOUT_SEC = 30
log = logging.getLogger(__name__)


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


SYSTEM_PROMPT = """You are the router for a Ford company assistant.
Pick exactly one path:
  - "rag": use Ford company/domain documents (policies, offers, vehicle info, support, history).
  - "sql": pricing/vehicle-model/dealer questions requiring structured data from the Ford demo SQLite DB.
  - "general": only if the question is clearly outside the Ford domain or purely generic knowledge.

Bias toward "rag" for anything mentioning Ford, vehicles, policies, warranties, returns, offers,
locations, history, or support topics. Bias toward "sql" for prices, cheapest/most expensive vehicles,
counts, or aggregations. Return strict JSON with keys decision, confidence, rationale. Confidence is
between 0 and 1.

Examples:
Q: "What is Ford's return policy on the Mustang?"       -> rag, 0.95
Q: "Tell me about Ford's history and founding year."    -> rag, 0.90
Q: "What does Ford say about EV battery warranties?"    -> rag, 0.91
Q: "What is the MSRP of the cheapest F-150 trim?"       -> sql, 0.92
Q: "How many Explorer vehicles were sold last year?"    -> sql, 0.88
Q: "Which dealer in Texas has the lowest F-150 price?"  -> sql, 0.85
Q: "What's the weather like today?"                     -> general, 0.97
Q: "How do I reset my iPhone?"                          -> general, 0.95"""


def _default_llm(model: str = DEFAULT_MODEL):
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai is not available")
    return ChatOpenAI(model=model, temperature=0.0, timeout=DEFAULT_TIMEOUT_SEC, max_retries=0)


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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    json_llm = (
        llm.bind(response_format={"type": "json_object"})
        if hasattr(llm, "bind")
        else llm
    )
    try:
        response = invoke_with_retry(
            json_llm,
            messages,
            timeout_sec=DEFAULT_TIMEOUT_SEC,
            max_attempts=3,
        )
    except RuntimeError as exc:
        log.warning("[judge] LLM unavailable, falling back to heuristic: %s", exc)
        fallback = _heuristic_judgement(prompt)
        fallback.raw_response = str(exc)
        return fallback

    raw_content: str = response.content if hasattr(response, "content") else str(response)

    try:
        data = json.loads(raw_content)
        parsed = VerdictSchema.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        log.warning(
            "[judge] JSON parse failed, falling back to heuristic. raw=%s",
            raw_content[:200],
        )
        fallback = _heuristic_judgement(prompt)
        fallback.raw_response = raw_content
        return fallback

    return JudgeVerdict(
        decision=parsed.decision,
        confidence=parsed.confidence,
        rationale=parsed.rationale,
        raw_response=raw_content,
    )


KEYWORDS_SQL = {
    "sql",
    "database",
    "table",
    "select",
    "query",
    "queries",
    "schema",
    "price",
    "pricing",
    "cheapest",
    "expensive",
    "cost",
    "sales",
    "sold",
    "revenue",
    "average",
    "sum",
    "count",
    "total",
    "how many",
    "how much",
    "aggregate",
    "units",
    "volume",
    "total sales",
    "best selling",
    "top selling",
}

# Domain cues that should route to the Ford document store.
KEYWORDS_RAG = {
    "document",
    "docs",
    "policy",
    "policies",
    "report",
    "manual",
    "vector",
    "embed",
    "warranty",
    "return",
    "returns",
    "refund",
    "support",
    "service",
    "offer",
    "offers",
    "discount",
    "student",
    "company",
    "history",
}

BRAND_TERMS = {
    "ford",
    "ford motor",
}

FORD_MODELS = [
    "mustang",
    "f-150",
    "bronco",
    "explorer",
    "maverick",
    "lincoln",
    "expedition",
    "escape",
    "edge",
]
BRAND_TERMS.update(FORD_MODELS)


def _heuristic_judgement(prompt: str) -> JudgeVerdict:
    lower = prompt.lower()

    if any(keyword in lower for keyword in KEYWORDS_SQL):
        decision = Decision.SQL
        rationale = "Detected SQL-oriented keywords."
    elif any(term in lower for term in BRAND_TERMS):
        decision = Decision.RAG
        rationale = "Detected Ford-specific terms; prefer document store."
    elif any(keyword in lower for keyword in KEYWORDS_RAG):
        decision = Decision.RAG
        rationale = "Detected Ford domain or document-related keywords."
    else:
        decision = Decision.GENERAL
        rationale = "Defaulting to general reasoning path."

    return JudgeVerdict(decision=decision, confidence=0.4, rationale=rationale)


__all__ = [
    "Decision",
    "JudgeVerdict",
    "FORD_MODELS",
    "judge_prompt",
]
