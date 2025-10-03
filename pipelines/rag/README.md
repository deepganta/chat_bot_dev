
This repository contains a modular ingestion pipeline designed for **Retrieval-Augmented Generation (RAG)** systems. It focuses on scraping / loading data, chunking, embedding via OpenAI, and storing vectors into a persistent vector database (Chroma).  

The pipeline is **local-first** but can be extended to cloud vector stores (Pinecone, Weaviate, Qdrant) later. It is built around **LangChain**, **LangGraph**, and **OpenAI embeddings**.

---

## 📂 Project Structure

```
pipelines/rag/ingestion/
│
├── pipeline.py        # CLI entrypoint to run the ingestion graph
├── graph_ingest.py    # LangGraph wiring of the pipeline nodes
├── load_clean.py      # Loads cleaned documents from disk
├── chunk.py           # Splits docs into token-aware (or char-based) chunks
├── embed_openai.py    # Embeds chunks with OpenAI embeddings
├── store.py           # Stores embeddings into Chroma (or JSONL fallback)
├── qa.py              # QA script: query the built index with OpenAI LLM
```

---

## ⚙️ Programs & Their Roles

### 1. **`pipeline.py`**
- Orchestrator / CLI entrypoint.
- Reads `corpus.yaml` and compiles the ingestion **LangGraph**.
- Runs sequentially: `load → chunk → embed → store`.

### 2. **`graph_ingest.py`**
- Defines the **LangGraph pipeline**.
- Shared state (`IngestState`) passes between nodes.
- Nodes:
  - `node_load`: load cleaned text from disk.
  - `node_chunk`: split text into smaller chunks.
  - `node_embed`: generate embeddings.
  - `node_store`: save to Chroma or JSONL.

### 3. **`load_clean.py`**
- Loads already-cleaned documents from:
  - `data/clean/cleaned.jsonl` (preferred format).
  - OR plain `.txt` files.

### 4. **`chunk.py`**
- Splits long texts into chunks:
  - **Preferred**: token-aware splitting with `tiktoken` (OpenAI tokenizer).
  - **Fallback**: character-based splitting (if `tiktoken` fails to build).

### 5. **`embed_openai.py`**
- Calls **OpenAI embeddings API** via `langchain-openai`.
- Default: `text-embedding-3-small` (1536 dims).

### 6. **`store.py`**
- Persists vectors into a **Chroma** collection (via `langchain-chroma`).
- If Chroma fails, falls back to writing JSONL (`vectors.jsonl`).

### 7. **`qa.py`**
- QA utility script.
- Loads the built index and creates a **Retriever + LLM** pipeline.

---

## ⚠️ Common Issues & Fixes

### Module Import Errors
- **Error:** `ModuleNotFoundError: No module named 'ingestion'`  
- **Fix:** Run as a module from project root:
  ```bash
  python -m pipelines.rag.ingestion.pipeline --config Configs/corpus.yaml
  ```

### tiktoken build error (Rust missing)
- **Error:** `can't find Rust compiler`  
- **Fix Options:**  
  - Remove `tiktoken` and use fallback.  
  - Install Rust.  
  - Use Python 3.11.

### Chroma deprecation warnings
- **Fix:** Install `langchain-chroma` and import as:
  ```python
  from langchain_chroma import Chroma
  ```

### Stored: 0 (empty embeddings)
- **Fix:** Ensure cleaned data exists under `<output_dir>/clean/`.

### QA returning "I don’t know"
- **Fix:** Confirm `corpus.yaml` matches the index, embeddings, and collection.

---

## 🚦 Running the Pipeline

### Ingest Ford data
```bash
export OPENAI_API_KEY=sk-...
python -m pipelines.rag.ingestion.pipeline --config Configs/corpus.yaml
```

### Query with QA
```bash
python -m pipelines.rag.ingestion.qa   --config Configs/corpus.yaml   --query "What does Ford say about global operations?"   --k 4   --model gpt-4o-mini
```

---

## 📖 Summary

This pipeline demonstrates:
- Modular ingestion architecture using **LangGraph**.
- Token/char-based chunking → OpenAI embeddings → Chroma storage.
- Config-driven design (`corpus.yaml`).
- QA interface with retriever + LLM.
- Documentation of issues faced and their fixes for traceability.

---

---

## 🏎️ Ford-Specific Setup

If you’re indexing **Ford** sources, use a dedicated output directory and collection to avoid mixing with other projects.

### 1) Example `corpus.yaml` (Ford)
```yaml
project_name: "ford-corp"
output_dir: "data_ford"
user_agent: "RAGScraper/1.0 (+https://example.com/contact)"
concurrency: 8
request_timeout_sec: 20
retry_attempts: 3
sleep_min_ms: 200
sleep_max_ms: 600
respect_robots: true

seed_urls:
  - "https://corporate.ford.com/operations.html"
  - "https://media.ford.com/content/fordmedia/fna/us/en/news.html"
  - "https://www.ford.com/support/"
  - "https://corporate.ford.com/company.html"
  - "https://www.ford.com/vehicles/"

# chunking (fallbacks to char-based if tiktoken unavailable)
chunk_max_tokens: 350
chunk_overlap_tokens: 40

# embeddings
embedding_model: "text-embedding-3-small"
embedding_dimensions: 1536

# vector store
collection_name: "ford-rag"
```

### 2) Provide cleaned inputs
The loader expects **either** `data_ford/clean/cleaned.jsonl` **or** `data_ford/clean/*.txt`.

Minimal `data_ford/clean/cleaned.jsonl` to smoke the pipeline:
```jsonl
{"url":"https://corporate.ford.com/operations.html","text":"Ford’s global operations span manufacturing, mobility, and electrification initiatives across regions.","meta":{"section":"operations"}}
{"url":"https://www.ford.com/support/","text":"Find owner manuals, maintenance schedules, and service support for Ford vehicles.","meta":{"section":"support"}}
{"url":"https://corporate.ford.com/company.html","text":"Ford Motor Company was founded in 1903 by Henry Ford and is a global automotive leader.","meta":{"section":"company"}}
```

### 3) Ingest
```bash
export OPENAI_API_KEY=sk-...
python -m pipelines.rag.ingestion.pipeline --config Configs/corpus.yaml
```

### 4) QA over the Ford index
```bash
python -m pipelines.rag.ingestion.qa   --config Configs/corpus.yaml   --query "What does Ford say about global operations?"   --k 4   --model gpt-4o-mini
```

### 5) Common Ford-specific pitfalls
- **Empty store** (`stored: 0`): No files in `data_ford/clean/`. Add `cleaned.jsonl` or `.txt` files.
- **QA returns no sources**: Mismatch between `output_dir`/`collection_name` in `corpus.yaml` used for ingest vs QA; or embedding model/dims differ.
- **Chroma deprecation warning**: Use `from langchain_chroma import Chroma` and ensure `langchain-chroma` is installed.
```bash
pip install langchain-chroma
```
- **tiktoken build fails**: Remove `tiktoken` and rely on char-based chunking, install Rust, or use Python 3.11.
```bash
# remove tiktoken from requirements OR
curl https://sh.rustup.rs -sSf | sh
```

---
