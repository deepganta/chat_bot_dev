# pipelines/rag/ingestion/qa.py
import os
import argparse
import yaml

# Try new package first, fall back to legacy import
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_vectorstore(cfg):
    idx_dir = f"{cfg.get('output_dir','data')}/index"
    coll    = cfg.get("collection_name", "rag")

    emb = OpenAIEmbeddings(
        model=cfg.get("embedding_model","text-embedding-3-small"),
        dimensions=cfg.get("embedding_dimensions",1536),
    )
    vs = Chroma(collection_name=coll, persist_directory=idx_dir, embedding_function=emb)
    return vs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to corpus.yaml")
    ap.add_argument("--query", required=True, help="Your question")
    ap.add_argument("--k", type=int, default=None, help="Top-K docs for retriever")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model for answer")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    cfg = load_cfg(args.config)
    k = args.k or cfg.get("retriever_k", 4)

    vs = build_vectorstore(cfg)

    # --- DEBUG: show how many docs in the collection
    try:
        count = vs._collection.count()  # works for both imports
    except Exception:
        # best-effort fallback if internal API changes
        count = -1
    print(f"[debug] collection='{cfg.get('collection_name','rag')}', index='{cfg.get('output_dir','data')}/index', docs={count}")

    # If no docs, bail early with a helpful message
    if count == 0:
        print("No documents in the vector store. Re-run ingest with the SAME corpus.yaml used here.")
        return

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

    llm = ChatOpenAI(model=args.model, temperature=args.temperature)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    # Use the modern API (invoke) to avoid deprecation warnings
    res = qa.invoke({"query": args.query})
    answer = res["result"]
    sources = res.get("source_documents", [])

    # --- DEBUG: also show the raw retrieved docs if empty
    if not sources:
        print("[debug] No sources returned by chain; inspecting retriever directly...")
        docs = retriever.get_relevant_documents(args.query)
        print(f"[debug] retriever returned {len(docs)} docs")
        for i, d in enumerate(docs[:k], 1):
            print(f"  - {i}: {d.metadata.get('url','<no-url>')} | {d.metadata.get('chunk_id','')}")
        print()

    print("\n=== Answer ===\n")
    print((answer or "").strip() or "I don't know.")

    print("\n=== Sources ===")
    if not sources:
        print("(No sources returned)")
    else:
        for i, d in enumerate(sources, 1):
            url = d.metadata.get("url", "<no-url>")
            order = d.metadata.get("order", "-")
            print(f"[{i}] {url} (order={order})")
    print()

if __name__ == "__main__":
    main()
