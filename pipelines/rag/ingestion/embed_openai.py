# ingestion/embed_openai.py
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings

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
    vecs = embeddings.embed_documents(texts)  # batch embed
    out = []
    for c, v in zip(chunks, vecs):
        out.append({**c, "embedding": v})
    return out
