# ingestion/chunk.py
from typing import List, Dict
import uuid
import tiktoken

def token_chunks(text: str, max_tokens: int, overlap_tokens: int, encoder) -> List[str]:
    """
    Split a large text into token-aware chunks using the provided tokenizer.
    Uses sliding window with overlap to preserve context across boundaries.
    """
    ids = encoder.encode(text)     # tokenize to ids
    chunks: List[str] = []
    i, n = 0, len(ids)
    while i < n:
        j = min(i + max_tokens, n) # end index for this chunk
        piece = encoder.decode(ids[i:j]).strip()
        if piece:
            chunks.append(piece)
        # slide window forward with overlap to maintain context
        i = j - overlap_tokens if j - overlap_tokens > i else j
    return chunks

def chunk_documents(cleaned_docs: List[Dict], max_tokens: int = 350, overlap_tokens: int = 40) -> List[Dict]:
    """
    Convert cleaned documents into a list of chunk records:
      { chunk_id, url, text, order, meta }
    """
    enc = tiktoken.get_encoding("cl100k_base")  # tokenizer compatible with OpenAI models
    out: List[Dict] = []
    for doc in cleaned_docs:
        text = doc.get("text", "")
        url = doc.get("url", "")
        meta = doc.get("meta", {})
        parts = token_chunks(text, max_tokens, overlap_tokens, enc)
        for idx, p in enumerate(parts):
            out.append({
                "chunk_id": str(uuid.uuid4()),         # stable id for the chunk
                "url": url,                             # keep a pointer to source
                "text": p,                              # the chunk content
                "order": idx,                           # position within the doc
                "meta": {**meta, "source": url or meta.get("source_file", "")},
            })
    return out
