# graph_ingest.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

from .load_clean import load_cleaned
from .chunk import chunk_documents
from .embed_openai import embed_chunks_openai
from .store import store_to_chroma_langchain

class IngestState(TypedDict, total=False):
    """Shared state that flows through the graph."""
    config: Dict
    cleaned_docs: List[Dict]
    chunks: List[Dict]
    chunks_with_vecs: List[Dict]
    store_stats: Dict

def node_load(state: IngestState) -> IngestState:
    """Load cleaned outputs from disk (JSONL preferred, TXT fallback)."""
    out_dir = state["config"].get("output_dir", "data")
    state["cleaned_docs"] = load_cleaned(out_dir)
    return state

def node_chunk(state: IngestState) -> IngestState:
    """Token-aware chunking using the config sizes."""
    cfg = state["config"]
    max_tokens = cfg.get("chunk_max_tokens", 350)
    overlap = cfg.get("chunk_overlap_tokens", 40)
    state["chunks"] = chunk_documents(state.get("cleaned_docs", []), max_tokens, overlap)
    return state

def node_embed(state: IngestState) -> IngestState:
    """OpenAI embeddings for each chunk."""
    cfg = state["config"]
    model = cfg.get("embedding_model", "text-embedding-3-small")
    dims = cfg.get("embedding_dimensions", 1536)
    state["chunks_with_vecs"] = embed_chunks_openai(state.get("chunks", []), model=model, dimensions=dims)
    return state

def node_store(state: IngestState) -> IngestState:
    """Persist vectors (Chroma preferred; JSONL fallback)."""
    cfg = state["config"]
    idx_dir = f'{cfg.get("output_dir","data")}/index'
    stats = store_to_chroma_langchain(
        state.get("chunks_with_vecs", []),
        index_dir=idx_dir,
        collection_name=cfg.get("collection_name", "rag"),
        embedding_model=cfg.get("embedding_model"),
        embedding_dimensions=cfg.get("embedding_dimensions"),
    )
    state["store_stats"] = stats
    return state

def build_graph():
    """LangGraph DAG wiring: load → chunk → embed → store."""
    g = StateGraph(IngestState)
    g.add_node("load", node_load)
    g.add_node("chunk", node_chunk)
    g.add_node("embed", node_embed)
    g.add_node("store", node_store)

    g.set_entry_point("load")
    g.add_edge("load", "chunk")
    g.add_edge("chunk", "embed")
    g.add_edge("embed", "store")
    g.add_edge("store", END)
    return g
