# ingestion/store.py
import os, json
from typing import List, Dict, Optional

def _persist_jsonl(index_dir: str, rows: List[Dict], name: str = "vectors.jsonl"):
    """Fallback storage: write vectors + metadata into a JSONL file."""
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"store": "JSONL", "path": path, "stored": len(rows)}

def store_to_chroma_langchain(chunks_with_vecs: List[Dict], index_dir: str = "data/index",
                              collection_name: str = "rag",
                              embedding_model: Optional[str] = None,
                              embedding_dimensions: Optional[int] = None) -> Dict:
    """
    Persist vectors to a Chroma collection (LangChain wrapper).
    If Chroma or bindings are unavailable, fall back to JSONL.
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain.schema import Document

        os.makedirs(index_dir, exist_ok=True)
        # embeddings function is optional here because we already computed vectors.
        embeddings = None
        if embedding_model and embedding_dimensions:
            # If you want Chroma to compute embeddings internally instead of using precomputed vecs.
            embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=embedding_dimensions)

        # Build LC Document objects for metadata + text
        docs = [
            Document(
                page_content=rec["text"],
                metadata={
                    "chunk_id": rec["chunk_id"],
                    "url": rec.get("url", ""),
                    "order": rec.get("order", 0),
                    **(rec.get("meta") or {})
                }
            ) for rec in chunks_with_vecs
        ]

        vs = Chroma(collection_name=collection_name, persist_directory=index_dir, embedding_function=embeddings)
        # Use low-level add to push precomputed embeddings directly (skip re-embedding).
        ids = [r["chunk_id"] for r in chunks_with_vecs]
        embs = [r["embedding"] for r in chunks_with_vecs]
        metas = [d.metadata for d in docs]
        texts = [d.page_content for d in docs]

        # Note: _collection is the underlying client; allowed here to avoid recomputation.
        vs._collection.add(ids=ids, embeddings=embs, metadatas=metas, documents=texts)
        vs.persist()
        return {"store": "Chroma", "persist_directory": index_dir, "stored": len(ids)}
    except Exception as e:
        # Fallback: serialize everything so you can still build a retriever later if needed.
        rows = [{
            "id": r["chunk_id"],
            "vector": r["embedding"],
            "meta": {"url": r.get("url", ""), "order": r.get("order", 0), **(r.get("meta") or {})},
            "text": r["text"],
        } for r in chunks_with_vecs]
        info = _persist_jsonl(index_dir, rows)
        info["error"] = str(e)
        return info
