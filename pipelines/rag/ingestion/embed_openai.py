# ingestion/embed_openai.py
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings

BATCH_SIZE = 512

def embed_chunks_openai(chunks: List[Dict], model: str = "text-embedding-3-small", dimensions: int = 1536) -> List[Dict]:
    """
    Generate vector embeddings for chunk texts via OpenAI.
    Returns the original chunk dicts augmented with an "embedding" key.
    """
    if not chunks:
        return []
    # LangChain wrapper for OpenAI embeddings; reads OPENAI_API_KEY from env
    embeddings = OpenAIEmbeddings(model=model, dimensions=dimensions)
    texts = [c["text"] for c in chunks]
    vecs: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        vecs.extend(embeddings.embed_documents(texts[i : i + BATCH_SIZE]))

    if len(vecs) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(chunks)} vectors, got {len(vecs)}"
        )

    out = []
    for c, v in zip(chunks, vecs):
        out.append({**c, "embedding": v})
    return out
