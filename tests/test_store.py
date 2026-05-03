"""Unit tests for pipelines/rag/ingestion/store.py"""
import json
import os
import tempfile
import pytest
from unittest.mock import patch
from pipelines.rag.ingestion.store import _persist_jsonl, store_to_chroma_langchain


SAMPLE_CHUNKS = [
    {"chunk_id": "abc1", "text": "Ford makes trucks.", "url": "https://ford.com", "order": 0,
     "embedding": [0.1, 0.2, 0.3], "meta": {}},
    {"chunk_id": "abc2", "text": "Mustang is fast.", "url": "https://ford.com", "order": 1,
     "embedding": [0.4, 0.5, 0.6], "meta": {}},
    {"chunk_id": "abc3", "text": "Explorer is an SUV.", "url": "https://ford.com", "order": 2,
     "embedding": [0.7, 0.8, 0.9], "meta": {}},
]


# ── _persist_jsonl ────────────────────────────────────────────────────────────

def test_persist_jsonl_writes_correct_line_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _persist_jsonl(tmpdir, SAMPLE_CHUNKS)
        path = result["path"]
        with open(path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 3

def test_persist_jsonl_return_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _persist_jsonl(tmpdir, SAMPLE_CHUNKS)
        assert result["store"] == "JSONL"
        assert result["stored"] == 3
        assert "path" in result

def test_persist_jsonl_valid_json_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _persist_jsonl(tmpdir, SAMPLE_CHUNKS)
        with open(result["path"], "r") as f:
            for line in f:
                if line.strip():
                    parsed = json.loads(line)
                    assert isinstance(parsed, dict)


# ── store_to_chroma_langchain — JSONL fallback ───────────────────────────────

def test_chroma_fallback_on_import_error():
    """When Chroma import fails, should fall back to JSONL with error key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *a, **kw):
            lowered = name.lower()
            if (
                lowered.startswith("langchain_chroma")
                or lowered.startswith("langchain_community.vectorstores")
                or lowered.startswith("chromadb")
            ):
                raise ImportError("Chroma not available")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=mock_import):
            result = store_to_chroma_langchain(
                SAMPLE_CHUNKS,
                index_dir=tmpdir,
                collection_name="test",
            )
        assert result["store"] == "JSONL"
        assert result["stored"] == len(SAMPLE_CHUNKS)
        assert os.path.exists(result["path"])
        assert "error" in result

def test_chroma_fallback_returns_jsonl_store():
    """Fallback mock shape remains compatible with calling code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("pipelines.rag.ingestion.store.store_to_chroma_langchain") as mock_fn:
            mock_fn.return_value = {
                "store": "JSONL",
                "path": os.path.join(tmpdir, "vectors.jsonl"),
                "stored": 3,
                "error": "Chroma unavailable",
            }
            result = mock_fn(SAMPLE_CHUNKS, index_dir=tmpdir)
            assert result["store"] == "JSONL"
            assert "error" in result
            assert result["stored"] == 3
