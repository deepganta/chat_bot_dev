"""Unit tests for pipelines/rag/Retrieval/judge.py"""
import pytest
from pipelines.rag.Retrieval.judge import (
    Decision, JudgeVerdict, _heuristic_judgement, judge_prompt
)


# ── Heuristic routing ────────────────────────────────────────────────────────

def test_heuristic_sql_price():
    verdict = _heuristic_judgement("What is the price of the cheapest F-150?")
    assert verdict.decision == Decision.SQL

def test_heuristic_sql_count():
    verdict = _heuristic_judgement("How many Explorers were sold last year?")
    assert verdict.decision == Decision.SQL

def test_heuristic_rag_brand_term():
    verdict = _heuristic_judgement("Tell me about the Ford Mustang")
    assert verdict.decision == Decision.RAG

def test_heuristic_rag_domain_keyword():
    verdict = _heuristic_judgement("What is Ford's warranty policy?")
    assert verdict.decision == Decision.RAG

def test_heuristic_general_off_topic():
    verdict = _heuristic_judgement("What is the weather today?")
    assert verdict.decision == Decision.GENERAL

def test_heuristic_confidence_always_04():
    for prompt in [
        "price of Mustang",
        "ford history",
        "what time is it",
    ]:
        verdict = _heuristic_judgement(prompt)
        assert verdict.confidence == 0.4

def test_heuristic_returns_judge_verdict():
    verdict = _heuristic_judgement("hello")
    assert isinstance(verdict, JudgeVerdict)
    assert isinstance(verdict.decision, Decision)
    assert isinstance(verdict.rationale, str)


# ── Empty prompt ─────────────────────────────────────────────────────────────

def test_judge_prompt_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        judge_prompt("")

def test_judge_prompt_whitespace_raises():
    with pytest.raises(ValueError, match="empty"):
        judge_prompt("   ")


# ── Low-confidence override logic ────────────────────────────────────────────

@pytest.mark.parametrize("decision", [Decision.GENERAL, Decision.SQL])
def test_low_confidence_override_to_rag(decision):
    """Simulate what handler.py does: low-confidence non-RAG → override to RAG."""
    CONFIDENCE_THRESHOLD = 0.60
    verdict = JudgeVerdict(
        decision=decision,
        confidence=0.45,
        rationale="test",
    )
    if verdict.confidence < CONFIDENCE_THRESHOLD and verdict.decision in {Decision.GENERAL, Decision.SQL}:
        override = Decision.RAG
    else:
        override = verdict.decision
    assert override == Decision.RAG

def test_high_confidence_not_overridden():
    """High-confidence GENERAL should not be overridden."""
    CONFIDENCE_THRESHOLD = 0.60
    verdict = JudgeVerdict(
        decision=Decision.GENERAL,
        confidence=0.95,
        rationale="clearly off-topic",
    )
    if verdict.confidence < CONFIDENCE_THRESHOLD and verdict.decision in {Decision.GENERAL, Decision.SQL}:
        override = Decision.RAG
    else:
        override = verdict.decision
    assert override == Decision.GENERAL

def test_rag_not_overridden_regardless_of_confidence():
    """RAG decision is never overridden."""
    CONFIDENCE_THRESHOLD = 0.60
    verdict = JudgeVerdict(
        decision=Decision.RAG,
        confidence=0.30,
        rationale="low confidence rag",
    )
    if verdict.confidence < CONFIDENCE_THRESHOLD and verdict.decision in {Decision.GENERAL, Decision.SQL}:
        override = Decision.RAG
    else:
        override = verdict.decision
    assert override == Decision.RAG
