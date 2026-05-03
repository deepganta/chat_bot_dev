"""Unit tests for pipelines/rag/ingestion/chunk.py"""
import pytest
from pipelines.rag.ingestion.chunk import chunk_documents


SAMPLE_DOC = {"url": "https://ford.com", "text": "Ford Motor Company builds great vehicles.", "meta": {"section": "home"}}


# ── Empty input ───────────────────────────────────────────────────────────────

def test_empty_list_returns_empty():
    assert chunk_documents([]) == []


# ── Single short document ─────────────────────────────────────────────────────

def test_short_doc_produces_at_least_one_chunk():
    chunks = chunk_documents([SAMPLE_DOC])
    assert len(chunks) >= 1

def test_chunk_has_required_keys():
    chunks = chunk_documents([SAMPLE_DOC])
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "url" in chunk
        assert "text" in chunk
        assert "order" in chunk
        assert "meta" in chunk

def test_chunk_id_is_nonempty_string():
    chunks = chunk_documents([SAMPLE_DOC])
    for chunk in chunks:
        assert isinstance(chunk["chunk_id"], str)
        assert len(chunk["chunk_id"]) > 0

def test_url_preserved():
    chunks = chunk_documents([SAMPLE_DOC])
    for chunk in chunks:
        assert chunk["url"] == "https://ford.com"


# ── Long document → multiple chunks ──────────────────────────────────────────

def test_long_doc_produces_multiple_chunks():
    long_text = "Ford makes great vehicles. " * 200
    doc = {"url": "https://ford.com/long", "text": long_text, "meta": {}}
    chunks = chunk_documents([doc], max_tokens=50, overlap_tokens=5)
    assert len(chunks) > 1

def test_order_starts_at_zero_and_increments():
    long_text = "Ford makes great vehicles. " * 200
    doc = {"url": "https://ford.com/long", "text": long_text, "meta": {}}
    chunks = chunk_documents([doc], max_tokens=50, overlap_tokens=5)
    orders = [c["order"] for c in chunks]
    assert orders[0] == 0
    assert orders == list(range(len(chunks)))


# ── Multiple documents ────────────────────────────────────────────────────────

def test_multiple_docs_all_chunked():
    docs = [
        {"url": "https://ford.com/1", "text": "Ford is great.", "meta": {}},
        {"url": "https://ford.com/2", "text": "Mustang is fast.", "meta": {}},
    ]
    chunks = chunk_documents(docs)
    urls = {c["url"] for c in chunks}
    assert "https://ford.com/1" in urls
    assert "https://ford.com/2" in urls
