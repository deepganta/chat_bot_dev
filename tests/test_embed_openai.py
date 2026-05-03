"""Unit tests for pipelines/rag/ingestion/embed_openai.py"""
import pytest
from unittest.mock import MagicMock, patch, call
from pipelines.rag.ingestion.embed_openai import embed_chunks_openai, BATCH_SIZE


SAMPLE_CHUNKS = [
    {"chunk_id": f"id{i}", "text": f"chunk text {i}", "url": "https://ford.com", "order": i, "meta": {}}
    for i in range(10)
]


# ── Empty input ────────────────────────────────────────────────────────────────

def test_empty_input_returns_empty():
    result = embed_chunks_openai([])
    assert result == []


# ── Output structure ───────────────────────────────────────────────────────────

def test_output_length_matches_input():
    fake_vector = [0.1] * 1536
    with patch("pipelines.rag.ingestion.embed_openai.OpenAIEmbeddings") as MockEmb:
        instance = MockEmb.return_value
        instance.embed_documents.return_value = [fake_vector] * len(SAMPLE_CHUNKS)
        result = embed_chunks_openai(SAMPLE_CHUNKS)
    assert len(result) == len(SAMPLE_CHUNKS)

def test_output_has_embedding_key():
    fake_vector = [0.1] * 1536
    with patch("pipelines.rag.ingestion.embed_openai.OpenAIEmbeddings") as MockEmb:
        instance = MockEmb.return_value
        instance.embed_documents.return_value = [fake_vector] * len(SAMPLE_CHUNKS)
        result = embed_chunks_openai(SAMPLE_CHUNKS)
    for item in result:
        assert "embedding" in item
        assert item["embedding"] == fake_vector

def test_original_keys_preserved():
    fake_vector = [0.5] * 10
    chunks = [{"chunk_id": "x1", "text": "hello", "url": "u", "order": 0, "meta": {}}]
    with patch("pipelines.rag.ingestion.embed_openai.OpenAIEmbeddings") as MockEmb:
        instance = MockEmb.return_value
        instance.embed_documents.return_value = [fake_vector]
        result = embed_chunks_openai(chunks)
    assert result[0]["chunk_id"] == "x1"
    assert result[0]["text"] == "hello"


# ── Batching ───────────────────────────────────────────────────────────────────

def test_600_items_triggers_two_embed_calls():
    chunks_600 = [
        {"chunk_id": f"id{i}", "text": f"text {i}", "url": "", "order": i, "meta": {}}
        for i in range(600)
    ]
    fake_vector = [0.1] * 5
    with patch("pipelines.rag.ingestion.embed_openai.OpenAIEmbeddings") as MockEmb:
        instance = MockEmb.return_value
        # First batch = 512, second batch = 88
        instance.embed_documents.side_effect = [
            [fake_vector] * BATCH_SIZE,
            [fake_vector] * (600 - BATCH_SIZE),
        ]
        result = embed_chunks_openai(chunks_600)
    assert instance.embed_documents.call_count == 2
    assert len(result) == 600


# ── Mismatch guard ─────────────────────────────────────────────────────────────

def test_mismatch_guard_raises_runtime_error():
    with patch("pipelines.rag.ingestion.embed_openai.OpenAIEmbeddings") as MockEmb:
        instance = MockEmb.return_value
        # Return fewer vectors than chunks
        instance.embed_documents.return_value = [[0.1] * 5] * (len(SAMPLE_CHUNKS) - 2)
        with pytest.raises(RuntimeError, match="[Mm]ismatch"):
            embed_chunks_openai(SAMPLE_CHUNKS)
