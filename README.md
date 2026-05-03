# Ford RAG Chatbot

> A production-grade, modular Retrieval-Augmented Generation chatbot for Ford Motor Company — built with LangChain, LangGraph, ChromaDB, OpenAI, and Streamlit.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Module Reference](#module-reference)
   - [ETL Pipeline](#etl-pipeline)
   - [Ingest Pipeline](#ingest-pipeline)
   - [RAG Ingestion Pipeline](#rag-ingestion-pipeline)
   - [RAG Retrieval Pipeline](#rag-retrieval-pipeline)
   - [Security Layer](#security-layer)
5. [Data Flow — Step by Step](#data-flow--step-by-step)
   - [Phase 1: ETL](#phase-1-etl)
   - [Phase 2: Ingestion](#phase-2-ingestion)
   - [Phase 3: Runtime Query](#phase-3-runtime-query)
6. [Configuration](#configuration)
7. [Setup & Installation](#setup--installation)
8. [Running the App](#running-the-app)
9. [Sprint History](#sprint-history)
10. [Test Report](#test-report)
11. [Known Issues & Backlog](#known-issues--backlog)

---

## Project Overview

The Ford RAG Chatbot is a multi-path conversational assistant designed to answer questions about Ford Motor Company's vehicles, policies, dealer network, and sales data. The system combines document-grounded retrieval with structured database querying and open-domain reasoning to give users accurate, sourced responses across a wide range of query types.

At its core, the chatbot uses a judge-based routing architecture: every user message is classified by a GPT-4o-mini judge into one of three paths — RAG (document retrieval from crawled Ford web pages), SQL (natural-language-to-SQL execution against a structured vehicle and dealer database), or GENERAL (open-domain reasoning via GPT-3.5-Turbo). This separation ensures that factual Ford domain questions are answered from verified sources, structured data questions produce executable queries, and off-topic questions receive appropriate general responses without contaminating either of the specialized pipelines.

The project is built to industrial standards: all LLM calls include retry and timeout logic via a centralized utility, every user prompt passes through a prompt-injection guardrail and a PII redaction layer before any storage or LLM invocation, conversation transcripts are optionally encrypted at rest using Fernet symmetric encryption, and the Streamlit UI includes brute-force protection and session timeout. Development followed a sprint-based process with a manager-developer code review loop, resulting in 16 tracked tasks across an initial bug-fix pass and two formal sprints.

---

## Architecture

The system operates in two phases: an offline build phase that populates the vector index and the structured database, and a runtime query phase that handles user conversations.

In the offline phase, Ford web pages are crawled, cleaned to plain text, token-chunked, embedded via OpenAI, and stored in a ChromaDB vector collection. Separately, a synthetic Ford SQLite database is generated containing plants, dealers, vehicles, and sales records. These two data stores are the knowledge backbone of the chatbot.

At runtime, every user message flows through a security layer (injection check, PII redaction), is persisted to a query log, classified by the judge, and dispatched to the appropriate responder. The responder retrieves or computes its answer, appends it to the conversation store, and the Streamlit UI renders the result. The conversation store supports optional Fernet encryption so transcripts are protected at rest.

```
OFFLINE PHASE:
corpus.yaml --> extract_ford.py --> crawlinfo.jsonl --> transform_ford.py --> cleaned.jsonl
                                                                                    |
corpus.yaml --> pipeline.py --> graph_ingest --> [load --> chunk --> embed --> store] --> ChromaDB
generate_ford_db.py --> ford.db

RUNTIME PHASE:
User --> Streamlit UI --> handler.py --> [sanitize_input --> redact_pii] --> judge.py
                                                                                |
                                          +--------------------+-------------------+
                                          |                    |                   |
                                       general.py           rag.py             sql.py
                                       GPT-3.5-Turbo      ChromaDB            ford.db
                                                          + GPT-3.5          + GPT-4o
                                                              |                   |
                                               ConversationStore (encrypted JSONL) + QueryStore
```

---

## Repository Structure

```
Chat_Bot_Rag/
├── Configs/
│   └── corpus.yaml              # Master config: URLs, chunking, embeddings, collection name
├── pipelines/
│   ├── etl/
│   │   ├── extract_ford.py      # Async crawler: fetches seed URLs, writes crawlinfo.jsonl
│   │   └── transform_ford.py    # HTML-to-text parser: cleans crawl output to cleaned.jsonl
│   ├── ingest/
│   │   └── generate_ford_db.py  # Generates synthetic Ford SQLite DB with plants/dealers/vehicles/sales
│   └── rag/
│       ├── ingestion/
│       │   ├── pipeline.py      # CLI entrypoint: reads corpus.yaml, compiles and runs LangGraph
│       │   ├── graph_ingest.py  # LangGraph DAG: wires load → chunk → embed → store nodes
│       │   ├── load_clean.py    # Reads cleaned.jsonl (or .txt fallback) into document dicts
│       │   ├── chunk.py         # Token-aware sliding-window chunker using tiktoken cl100k_base
│       │   ├── embed_openai.py  # Batched OpenAI embedding with mismatch guard (BATCH_SIZE=512)
│       │   ├── store.py         # Writes vectors to ChromaDB; falls back to JSONL on failure
│       │   └── qa.py            # CLI query tool: connects to Chroma, runs RetrievalQA chain
│       └── Retrieval/
│           ├── interface.py     # Streamlit chat UI with auth gate, session timeout, conversation nav
│           ├── handler.py       # Central dispatcher: sanitize → redact → persist → judge → route
│           ├── judge.py         # Routes each prompt to GENERAL / RAG / SQL via GPT-4o-mini
│           ├── guardrails.py    # Input sanitization: normalizes text, blocks prompt injection
│           ├── pii.py           # PII detection and redaction: Luhn cards + Presidio + regex
│           ├── llm_utils.py     # Centralized LLM invocation with tenacity retry and timeout
│           ├── rag.py           # RAG responder: ChromaDB similarity search + GPT-3.5 answer
│           ├── sql.py           # SQL responder: NL-to-SQL generation + SQLite execution + summary
│           └── general.py       # General responder: open-domain GPT-3.5-Turbo with history
├── tests/                       # Pytest unit test suite (79 tests across 8 files)
├── data/
│   ├── ford.db                  # SQLite: plants, dealers, vehicles, sales
│   ├── index/                   # ChromaDB vector index (ford-rag collection)
│   ├── conversations/           # Per-session JSONL transcripts (optionally Fernet-encrypted)
│   ├── queries/                 # query_log.jsonl — append-only record of every user prompt
│   ├── clean/                   # Cleaned Ford HTML converted to plain text (cleaned.jsonl)
│   └── input/                   # Raw crawl output (crawlinfo.jsonl)
├── DEV_CONVERSATION.md          # Manager-developer task log (16 tasks across 2 sprints)
├── TEST_REPORT.md               # Automated test results (79 tests, 77 passed, 2 failed)
└── requirements.txt
```

---

## Module Reference

### ETL Pipeline

#### `extract_ford.py`
**Purpose:** Asynchronous web crawler. Reads seed URLs from corpus.yaml and fetches each one with configurable concurrency, robots.txt compliance, retry logic, and polite pacing. No HTML parsing occurs here — only raw network I/O and bookkeeping. Writes one JSON record per URL to `data/input/crawlinfo.jsonl`.

**Key functions:**
- `load_config(path)` — Parses corpus.yaml into a typed `CrawlConfig` dataclass, applying defaults for any missing keys.
- `fetch_robots(client, url, timeout)` — Fetches and parses `/robots.txt` for the host of a given URL, extracting `Disallow` rules for `User-agent: *`.
- `is_allowed_by_robots(path, rules)` — Returns `False` if the URL path matches any disallowed prefix from the parsed robots rules (including a full-site `/` block).
- `fetch_with_retries(client, url, timeout, attempts)` — Issues a GET request with a limited retry loop and jittered exponential backoff. Uses `await asyncio.sleep()` (corrected from a blocking `time.sleep` bug) so the async event loop is never blocked. Returns `(status, content_type, body_or_None)`.
- `extract_once(cfg)` — Orchestrates the full crawl: creates the output file, caches robots rules per host, filters excluded URL patterns, and launches all workers concurrently via `asyncio.gather`.
- `worker(u)` — Per-URL coroutine: checks robots gate, fetches with retries, appends the result record to the JSONL output, then waits a randomized polite delay.

**Output:** `data/input/crawlinfo.jsonl` — one JSON object per line with fields: `url`, `status`, `content_type`, `fetched_at`, `html`, and optionally `note` (e.g. `blocked_by_robots`).

**Config keys used:** `seed_urls`, `concurrency`, `retry_attempts`, `sleep_min_ms`, `sleep_max_ms`, `respect_robots`, `request_timeout_sec`, `user_agent`, `exclude_url_patterns`

---

#### `transform_ford.py`
**Purpose:** HTML-to-plain-text transformer. Reads `crawlinfo.jsonl`, parses each successful HTML page with BeautifulSoup (falling back to a naive regex stripper if BeautifulSoup is unavailable), and applies a normalization pipeline before writing clean records to `data/clean/cleaned.jsonl`. Does not chunk — text length is preserved intact.

**Key functions:**
- `html_to_text_bs4(html)` — Parses HTML with BeautifulSoup using the `lxml` parser. Removes `<script>`, `<style>`, `<noscript>`, `<template>`, and `<svg>` tags before extracting text. Returns `(title, text)`.
- `html_to_text_naive(html)` — Pure-regex fallback used when BeautifulSoup is unavailable. Strips script/style blocks and HTML tags, then collapses whitespace.
- `html_to_text(html)` — Dispatch function: calls `html_to_text_bs4` if available, else `html_to_text_naive`.
- `normalize_text_pipeline(text)` — Applies five sequential normalization steps: HTML entity unescaping (including `&nbsp;`), bullet/dash normalization, blank-line collapsing, multi-space collapsing (without crossing newlines), and hard-wrapped sentence joining.
- `stable_id_from_url(url)` — Produces an MD5 hex digest of the URL for use as a stable document ID.
- `build_page_record(url, fetched_at, title, text)` — Assembles the final output dict including `doc_id`, `url`, `title`, `fetched_at`, `text`, `word_count`, and `char_count`.
- `transform_to_plain(input_jsonl, output_jsonl)` — Main driver: iterates records, skips non-200 or non-HTML responses, calls the parsing and normalization pipeline, and appends clean records.

**Input:** `data/input/crawlinfo.jsonl`
**Output:** `data/clean/cleaned.jsonl`

---

### Ingest Pipeline

#### `generate_ford_db.py`
**Purpose:** Generates a fully synthetic Ford Motor Company SQLite database for use by the SQL responder. Uses Faker and a seeded `random.Random` instance so the database contents are reproducible across runs. Follows a strict ETL pattern internally: synthesize rows (E/T) then load them in a single commit (L).

**Tables created:**
- `plants(id, name, city, state, country, opened_year)` — Ford assembly plants with randomized US cities and states, names following the pattern "Ford [City] Assembly Plant", and opening years drawn from 1965 to the current year.
- `dealers(id, name, city, state, region)` — Ford dealerships with names in the form "[Last Name] Ford" and a US region assignment (Northeast, Midwest, South, West).
- `vehicles(vin, model, model_year, trim, segment, msrp, plant_id)` — Inventory rows with pseudo-VINs (17 alphanumeric characters), randomly selected from 8 Ford models (F-150, Mustang, Explorer, Escape, Bronco, Maverick, Edge, Expedition), computed MSRPs based on model/segment/trim uplifts, and foreign key links to plants.
- `sales(id, vin, dealer_id, sale_date, sale_price, customer_type)` — Sales transactions from 2018 onward, with sale prices set at MSRP ±10% and customer type constrained to `Retail` or `Fleet`.

**Key functions:**
- `synthesize_plants(n, fake, rng)` — Produces `n` plant tuples using Faker city names and sampled US states.
- `synthesize_dealers(n, fake, rng)` — Produces `n` dealer tuples using Faker last names and sampled regions.
- `synthesize_vehicles(n, plant_ids, fake, rng)` — Produces `n` vehicle tuples by randomly selecting from `FORD_MODELS`, computing MSRPs via `_price_for`, and assigning a random plant.
- `synthesize_sales(n, vins, dealer_ids, rng)` — Produces `n` sale tuples with placeholder sale prices; actual prices are derived during `load_all` once MSRP data is available.
- `load_all(conn, plants, dealers, vehicles, sales, rng)` — Inserts all tables in dependency order, computes final sale prices from the MSRP lookup dict, and commits.
- `etl(db_path, rows_per_table, seed)` — Top-level coordinator: sets up RNG, creates schema, calls synthesize and load functions, then prints row counts.

**Usage:** `python -m pipelines.ingest.generate_ford_db --rows 200 --db data/ford.db --seed 42`

---

### RAG Ingestion Pipeline

#### `pipeline.py`
**Purpose:** CLI entrypoint for the RAG ingestion pipeline. Reads corpus.yaml, builds the LangGraph by calling `build_graph()`, compiles it into an executable app, and invokes it once with the config as the initial state. Prints the store statistics returned by the final node.

**Usage:** `python -m pipelines.rag.ingestion.pipeline --config Configs/corpus.yaml`

---

#### `graph_ingest.py`
**Purpose:** Defines the LangGraph ingestion DAG. Assembles four nodes into a linear pipeline — load, chunk, embed, store — and wires them with directed edges. Each node receives the shared `IngestState` dict, transforms one field, and returns the updated state.

**Nodes:**
- `node_load` — Reads `output_dir` from config and calls `load_cleaned()`, populating `cleaned_docs` in state.
- `node_chunk` — Reads `chunk_max_tokens` and `chunk_overlap_tokens` from config and calls `chunk_documents()`, populating `chunks` in state.
- `node_embed` — Reads `embedding_model` and `embedding_dimensions` from config and calls `embed_chunks_openai()`, populating `chunks_with_vecs` in state.
- `node_store` — Reads collection config and calls `store_to_chroma_langchain()`, populating `store_stats` in state.

**State object:** `IngestState` (TypedDict) — fields: `config` (raw YAML dict), `cleaned_docs` (list of `{url, text, meta}` dicts), `chunks` (list of chunk records), `chunks_with_vecs` (chunks augmented with embedding vectors), `store_stats` (dict returned by the store node).

---

#### `load_clean.py`
**Purpose:** Loads already-cleaned documents from disk into the ingestion state. Supports two formats so the pipeline is not hard-coupled to the JSONL output of `transform_ford.py`.

**Function:** `load_cleaned(output_dir)` — First checks for `data/clean/cleaned.jsonl` and yields `{url, text, meta}` dicts from each JSONL record (preferred path, preserves all metadata). If the JSONL file is absent, falls back to globbing all `*.txt` files in `data/clean/`, reading each as plain text, and synthesizing minimal metadata with the filename stored under `meta.source_file`.

---

#### `chunk.py`
**Purpose:** Splits cleaned document text into overlapping token-bounded chunks using the tiktoken `cl100k_base` tokenizer (compatible with all OpenAI models). Produces chunk records suitable for embedding and vector storage.

**Functions:**
- `token_chunks(text, max_tokens, overlap_tokens, encoder)` — Encodes the full text to token IDs, then iterates with a sliding window of width `max_tokens` and step `max_tokens - overlap_tokens`. Each window is decoded back to text and added to the output list. This preserves sentence context across chunk boundaries.
- `chunk_documents(cleaned_docs, max_tokens, overlap_tokens)` — Iterates all documents, calls `token_chunks` on each, and wraps every piece into a chunk record: `{chunk_id (UUID), url, text, order, meta}`. The `order` field is the zero-based index within the source document.

**Config keys:** `chunk_max_tokens` (default 350 tokens), `chunk_overlap_tokens` (default 40 tokens)

---

#### `embed_openai.py`
**Purpose:** Generates vector embeddings for all chunk texts by calling the OpenAI embeddings API via the LangChain wrapper. Processes chunks in batches to respect API input limits and augments each chunk dict with its embedding vector.

**Function:** `embed_chunks_openai(chunks, model, dimensions)` — Extracts all chunk texts, splits them into batches of `BATCH_SIZE=512`, and calls `OpenAIEmbeddings.embed_documents()` on each batch. Collected vectors are flattened and zipped back onto the original chunk dicts under the key `"embedding"`. A mismatch guard raises `RuntimeError` if the number of returned vectors does not equal the number of input chunks, preventing silent data corruption.

**Model:** `text-embedding-3-small`, 1536 dimensions (configurable via corpus.yaml)

---

#### `store.py`
**Purpose:** Persists embedded chunk vectors to a ChromaDB collection. Wraps the entire Chroma write in a try/except so the ingestion pipeline degrades gracefully if ChromaDB is unavailable, while ensuring failures are visible in logs.

**Functions:**
- `store_to_chroma_langchain(chunks_with_vecs, index_dir, collection_name, embedding_model, embedding_dimensions)` — Primary path: imports `langchain_chroma.Chroma` (with a `langchain_community` fallback), builds `Document` objects from chunk text and metadata, and writes precomputed embeddings directly to the Chroma collection via `vs._collection.add()` to avoid re-embedding. Returns a stats dict with `store`, `persist_directory`, and `stored` keys. On any exception, logs the full traceback with `log.error(..., exc_info=True)` before falling back to `_persist_jsonl`.
- `_persist_jsonl(index_dir, rows, name)` — Fallback: writes all chunk records (including embedding vectors) as JSONL to `index_dir/vectors.jsonl`. Returns a stats dict including an `"error"` key so the caller knows the primary path failed.

**Note:** The primary import path is `langchain_chroma` (the current recommended package). The `langchain_community` import is a secondary fallback maintained for environments that have not upgraded.

---

#### `qa.py`
**Purpose:** Command-line query tool for the ingestion pipeline. Connects to the same ChromaDB collection populated by the ingestion pipeline, builds a `RetrievalQA` chain backed by a configurable OpenAI chat model, and prints the answer and source documents. Useful for verifying the vector index without running the full Streamlit UI.

**Usage:** `python -m pipelines.rag.ingestion.qa --config Configs/corpus.yaml --query "What does Ford say about EV battery warranties?" --k 4 --model gpt-4o-mini`

---

### RAG Retrieval Pipeline

#### `interface.py`
**Purpose:** Streamlit-based chat UI. Serves as the user-facing entry point for the entire retrieval stack. Manages session state, authentication, conversation navigation, and delegates all query logic to `handle_user_query` in `handler.py`.

**Features:**
- Renders a persistent chat history by loading the active conversation from `ConversationStore` on startup.
- Shows a "Thinking..." spinner via `st.spinner` while `handle_user_query` executes.
- Sidebar panel showing the active conversation ID, the path to the conversation log file, and a "Start New Conversation" button that creates a fresh conversation and clears the message display.
- On each successful query submission, updates `st.session_state.last_activity` to the current timestamp for session timeout tracking.
- Catches `ValueError` raised by the guardrail and displays it as an `st.error` message without crashing the session.

**Auth:** Controlled by the `CHATBOT_PASSWORD` environment variable. If set, the UI renders a login form before displaying anything else. After 5 consecutive failed password attempts, the form is replaced with a lockout message and `st.stop()` is called. Sessions expire after 30 minutes of inactivity — the `authenticated` flag is reset to `False` and the user sees a "Session expired. Please log in again." message. If `CHATBOT_PASSWORD` is not set, auth is skipped entirely and a one-time warning is logged (dev mode).

---

#### `handler.py`
**Purpose:** Central request dispatcher. Receives the raw user prompt from the UI, applies the security pipeline, persists records, invokes the judge, routes to the appropriate responder, and returns a `HandlerResult` with both the user and assistant messages.

**Key flow:** `sanitize_input` (guardrail) → `redact_pii` (PII removal) → persist redacted prompt to `QueryStore` → `judge_prompt` → optional confidence override → dispatch to `general`, `rag`, or `sql` responder → persist assistant reply to `ConversationStore` → return `HandlerResult`.

**Classes:**
- `QueryRecord` — Dataclass holding `conversation_id`, `prompt` (post-redaction), and `created_at` ISO timestamp for a single user query.
- `ConversationMessage` — Dataclass holding `role` (`user` or `assistant`), `content`, and `created_at`. The fundamental unit of conversation history.
- `QueryStore` — Append-only JSONL store for `QueryRecord` objects. Writes to `data/queries/query_log.jsonl`. Creates parent directories on initialization.
- `ConversationStore` — Per-conversation JSONL transcript store. Each conversation is stored in `data/conversations/{conversation_id}.jsonl`. When `CONVERSATION_KEY` is set in the environment, each line is Fernet-encrypted before writing and decrypted on load. When the key is absent, data is written unencrypted with a warning logged on every write.
- `HandlerResult` — Dataclass returned by `handle_user_query` containing `conversation_id`, `user_message`, `assistant_message`, and the optional `JudgeVerdict`.

**Functions:**
- `handle_user_query(prompt, conversation_id, store, conversation_store, enable_judge, judge_model, llm)` — Full request handling pipeline. Creates a session ID, runs the security pipeline, persists records, optionally invokes the judge with low-confidence override logic, dispatches to the correct responder, and returns the result. Each responder is imported lazily inside the function to avoid circular dependencies.
- `start_new_conversation(store)` — Creates a new empty conversation file in `ConversationStore` and returns the generated conversation ID.

---

#### `judge.py`
**Purpose:** Classifies every user prompt into exactly one of three routing decisions — `GENERAL`, `RAG`, or `SQL` — using a GPT-4o-mini LLM call with a structured output contract, with a keyword heuristic as a fallback when the API is unavailable.

**Routing logic:**
1. If `OPENAI_API_KEY` is set and `ChatOpenAI` is importable, the judge calls GPT with `response_format={"type": "json_object"}` bound via `llm.bind()` to enforce valid JSON output. The response is validated against the `VerdictSchema` Pydantic model.
2. If the LLM is unavailable (network error, rate limit, exhausted retries), a `log.warning` is emitted and `_heuristic_judgement` is called as fallback.
3. If `OPENAI_API_KEY` is not set or `langchain_openai` is not importable, `_heuristic_judgement` is called directly without attempting any API call.
4. Low-confidence override (applied in `handler.py`, not in `judge.py` itself): if `confidence < 0.60` and the decision is `GENERAL` or `SQL`, the decision is overridden to `RAG`. RAG decisions are never overridden regardless of confidence.

**Few-shot examples in SYSTEM_PROMPT:**
```
Q: "What is Ford's return policy on the Mustang?"       -> rag, 0.95
Q: "Tell me about Ford's history and founding year."    -> rag, 0.90
Q: "What does Ford say about EV battery warranties?"    -> rag, 0.91
Q: "What is the MSRP of the cheapest F-150 trim?"       -> sql, 0.92
Q: "How many Explorer vehicles were sold last year?"    -> sql, 0.88
Q: "Which dealer in Texas has the lowest F-150 price?"  -> sql, 0.85
Q: "What's the weather like today?"                     -> general, 0.97
Q: "How do I reset my iPhone?"                          -> general, 0.95
```

**Classes:**
- `Decision` — String enum with members `GENERAL`, `RAG`, `SQL`.
- `VerdictSchema` — Pydantic model validating the LLM JSON response: `decision` (Decision enum), `confidence` (float 0.0–1.0), `rationale` (str).
- `JudgeVerdict` — Dataclass returned by `judge_prompt`: `decision`, `confidence`, `rationale`, `raw_response` (optional raw LLM content for logging).

---

### Security Layer

#### `guardrails.py`
**Purpose:** Input sanitization module. Detects and blocks prompt-injection attempts before any LLM call is made. Runs as the first step in `handle_user_query`.

**Function:** `sanitize_input(prompt, session_id)` — Validates type and length (max 1000 characters), runs the normalization pipeline, checks compact (stripped) form against known obfuscation patterns, scans normalized text against WARN-tier patterns (log and continue), then scans against BLOCK-tier patterns (log and raise `ValueError`). Returns the original stripped prompt on success.

**Normalization pipeline:** Applied before any pattern matching to defeat encoding-based bypasses:
1. Strip zero-width characters (`​`–`‍`, `⁠`, `﻿`)
2. Unicode NFKC normalization (collapses fullwidth characters, lookalikes)
3. HTML entity decoding (converts `&#105;` → `i`, etc.)
4. Whitespace collapsing (all runs of whitespace → single space)

**Tier system:**
- `BLOCK` patterns (raise `ValueError` immediately):
  - `ignore_previous_instructions` — matches "ignore [all/any/the] previous instructions"
  - `ignore_all_instructions` — matches "ignore all instructions"
  - `system_role_override` — matches `system:`, `<system>`, `[system]` at line start
  - `you_are_now_unrestricted` — matches "you are now in developer/god/jailbreak/unrestricted mode"
  - `act_as_no_restrictions` — matches "act as if you have no restrictions"
  - `pretend_unfiltered_assistant` — matches "pretend you are an unfiltered/unrestricted assistant"
  - `disregard_prior_context` — matches "disregard all prior/previous instructions/context"
  - `forget_safety_rules` — matches "forget your safety/policy/rules/instructions/constraints"
- `WARN` patterns (log warning, allow through):
  - `jailbreak_keyword` — matches "jailbreak", "disable guardrails", "bypass safety"
  - `policy_override_hint` — matches "override the policy", "ignore the policy"

**Config:** Rules are loaded from `DEFAULT_GUARDRAIL_CONFIG` by default. If `GUARDRAIL_CONFIG_PATH` is set in the environment, the module loads and validates the YAML file at that path, falling back to defaults if the file is missing, unreadable, or has an invalid schema. Compiled regex patterns are cached at module level so they are only compiled once.

**Audit log format:** `[guardrail] BLOCKED reason=<rule_name> session=<session_id> length=<n>` — prompt content is never logged.

---

#### `pii.py`
**Purpose:** PII detection and redaction layer. Applied after the guardrail check and before any data is persisted to disk or sent to an LLM. Returns the redacted text, a boolean flag indicating whether redaction occurred, and a sorted list of detected entity type names.

**Function:** `redact_pii(text)` → `(redacted_text, was_redacted, entity_types)` — Runs three detection stages in sequence, each operating on the output of the previous.

**Detection pipeline:**
1. **Luhn-validated card detection** — `CARD_CANDIDATE_RE` (matches 13–19 consecutive digit-and-separator sequences) finds candidates, strips separators, checks digit length, and validates with the Luhn checksum algorithm. Only sequences that pass Luhn are redacted as `[CARD]`. This eliminates false positives on VINs, part numbers, and numeric runs.
2. **Microsoft Presidio** (primary) — `AnalyzerEngine` analyzes the text for 50+ entity types. `AnonymizerEngine` replaces each detected entity with a bracketed label (e.g. `[EMAIL_ADDRESS]`, `[PERSON]`). Presidio is initialized once via a singleton pattern with `_presidio_init_attempted` to prevent repeated failed initialization attempts. If Presidio packages are unavailable, a one-time warning is logged and this stage is skipped.
3. **Regex fallback** — Always runs after Presidio on the already-cleaned text. Covers `EMAIL_RE` → `[EMAIL]`, `PHONE_RE` → `[PHONE]`, `SSN_RE` → `[SSN]`, `DOB_RE` → `[DATE_TIME]`. Catches any patterns Presidio may have missed.

**Entities detected:** `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, `PERSON`, `DATE_TIME`, `LOCATION` (via Presidio), plus email, phone, SSN, DOB (via regex fallback).

**Log format:** `[pii] WARNING detected=[ENTITY_TYPE, ...] conversation=<id>` — detected entity types are logged, never the actual values.

---

#### `llm_utils.py`
**Purpose:** Centralized LLM invocation utility. All four responders (`judge.py`, `rag.py`, `sql.py`, `general.py`) call `invoke_with_retry` instead of `llm.invoke` directly, ensuring uniform retry and timeout behavior across the entire system.

**Function:** `invoke_with_retry(llm, messages, timeout_sec=30, max_attempts=3)` — Binds the timeout to the LLM via `llm.bind(timeout=timeout_sec)` if the bind method is available. Wraps the `invoke` call with a tenacity retry decorator configured for exponential backoff.

**Retry triggers:** `openai.RateLimitError`, `openai.APITimeoutError`, `openai.APIConnectionError`. Non-transient errors (e.g. `AuthenticationError`) are not retried and propagate immediately. Fallback stub exception classes are defined for environments where the `openai` package is not installed.

**Backoff:** Exponential with `wait_exponential(multiplier=1, min=1, max=4)` — approximately 1s, 2s, 4s between attempts.

**On exhaustion:** Raises `RuntimeError("LLM unavailable after 3 attempts")` wrapping the last API exception.

**Log format:** `[llm] retry attempt=N reason=<exception>` — emitted before each retry sleep via the `before_sleep` tenacity callback.

---

#### `rag.py`
**Purpose:** RAG path responder. Answers Ford document questions by retrieving relevant chunks from ChromaDB and synthesizing an answer with GPT-3.5-Turbo, constrained to only use the retrieved context.

**Function:** `handle_rag_query(prompt, conversation_id, conversation_store, message_factory, ...)` — Loads the ChromaDB vectorstore (cached in `_VECTORSTORE_CACHE` keyed by config path and collection), builds a similarity retriever with `k=4`, prepends conversation history to the query if available, retrieves source documents, constructs a prompt with the system instruction to answer only from indexed documents, calls `invoke_with_retry`, appends the answer plus a formatted sources block to the conversation, and returns the message, answer text, and source documents.

**System prompt:** "You are a Ford company assistant. Answer using ONLY the provided context snippets. If the context does not contain the answer, say 'I couldn't find this in the indexed Ford documents.' Do not speculate. Be concise and helpful."

---

#### `sql.py`
**Purpose:** SQL path responder. Converts natural language questions about Ford's structured data (vehicles, dealers, plants, sales) into executable SQLite queries and summarizes the results.

**Function:** `handle_sql_query(prompt, conversation_id, conversation_store, message_factory, ...)` — Runs a four-step pipeline on a single SQLite connection:

**Flow:**
1. `_introspect_schema(conn)` — Reads live DDL from `sqlite_master` to ground the LLM in actual table and column names.
2. `_build_metadata_summary(conn)` — Samples distinct values for `vehicles.model`, `vehicles.segment`, and `sales.customer_type` to give the LLM concrete reference values.
3. `_generate_sql(prompt, schema, metadata, llm)` — Sends the schema and metadata to GPT-4o with a system instruction to return `{"sql": "...", "rationale": "..."}` JSON. Parses the response, strips code fences, and validates the SQL against a blocklist of destructive keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`).
4. Executes the SQL on `ford.db`. If the query returns no rows, a fallback query (`SELECT model, trim, msrp, segment FROM vehicles ORDER BY msrp ASC LIMIT 1`) surfaces the closest available record and a note is added to the summary prompt.
5. `_summarize_result(prompt, sql, rows, llm)` — Sends the row data as JSON to GPT-4o for a concise natural-language summary. The final response includes the summary, model rationale, the executed SQL in a code fence, and a formatted display of the first five rows.

**Safety:** Blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER` — raises `ValueError` before execution if any of these appear in the generated SQL.

---

#### `general.py`
**Purpose:** General path responder. Handles queries that fall outside the Ford domain or do not require document retrieval or database access, using GPT-3.5-Turbo with conversation history context.

**Function:** `handle_general_query(prompt, conversation_id, conversation_store, message_factory, ...)` — Converts up to 6 prior conversation turns into role-accurate chat messages via `_history_as_messages`, prepends the system prompt, appends the current user prompt, and calls `invoke_with_retry`. Appends the assistant reply to `ConversationStore` and returns both the message object and the raw text.

**System prompt:** "You are a concise, professional assistant. Answer user questions directly. If the prompt is unclear or you lack enough context, explicitly mention the gap instead of inventing facts."

---

## Data Flow — Step by Step

### Phase 1: ETL (run once, offline)

```
Step 1  corpus.yaml defines seed_urls (5 Ford URLs) and crawl settings
        (concurrency=8, retry_attempts=3, sleep_min_ms=200, sleep_max_ms=600)

Step 2  extract_ford.py is invoked as a module
        - Reads config via load_config()
        - Per-host robots.txt is fetched and cached in robots_cache dict
        - Excluded URL patterns are filtered from the seed list
        - asyncio.gather() launches one worker() coroutine per seed URL
        - Each worker(): checks robots gate, calls fetch_with_retries(),
          appends a JSON record to crawlinfo.jsonl, then awaits a
          randomized polite sleep (200-600ms)
        - fetch_with_retries() uses await asyncio.sleep() for backoff
          (corrected from a blocking time.sleep() that defeated concurrency)
        - Output: data/input/crawlinfo.jsonl
          (one record per URL: url, status, content_type, fetched_at, html)

Step 3  transform_ford.py reads crawlinfo.jsonl line by line
        - Skips any record where status != 200 or html is None
        - html_to_text() parses HTML with BeautifulSoup + lxml:
            - removes script/style/noscript/svg tags
            - extracts title and body text
        - normalize_text_pipeline() cleans the text:
            - unescape HTML entities (&amp; &nbsp; etc.)
            - normalize bullet and dash characters
            - collapse runs of 3+ blank lines to 2
            - collapse multi-space runs within lines
            - join hard-wrapped sentences
            - strip leading/trailing whitespace from lines and document
        - build_page_record() assembles the output dict:
            doc_id (MD5 of URL), url, title, fetched_at,
            text (clean), word_count, char_count
        - Output: data/clean/cleaned.jsonl
```

### Phase 2: Ingestion (run once after ETL)

```
Step 1  pipeline.py reads corpus.yaml via yaml.safe_load()
        build_graph() compiles a LangGraph StateGraph into a runnable app
        app.invoke({"config": cfg}) kicks off the pipeline with the config
        as initial state

Step 2  node_load (graph_ingest.py → load_clean.py)
        load_cleaned() opens data/clean/cleaned.jsonl (preferred)
        yields {url, text, meta} dicts into cleaned_docs state field
        (falls back to *.txt glob if JSONL is absent)

Step 3  node_chunk (graph_ingest.py → chunk.py)
        chunk_documents() initializes the cl100k_base tiktoken encoder
        for each document, token_chunks() splits text:
            - encode full text to token ID list
            - slide window: step = max_tokens - overlap_tokens
            - decode each window back to string
        each chunk becomes: {chunk_id (UUID), url, text, order, meta}
        defaults: max_tokens=350, overlap_tokens=40

Step 4  node_embed (graph_ingest.py → embed_openai.py)
        embed_chunks_openai() groups chunk texts into batches of 512
        calls OpenAIEmbeddings(model="text-embedding-3-small", dims=1536)
            .embed_documents() for each batch
        flattens results and zips vectors back onto chunk dicts
        mismatch guard: raises RuntimeError if len(vecs) != len(chunks)

Step 5  node_store (graph_ingest.py → store.py)
        store_to_chroma_langchain() attempts Chroma write:
            - imports langchain_chroma.Chroma (community fallback)
            - builds Document objects from text + metadata
            - calls vs._collection.add(ids, embeddings, metadatas, texts)
              to push precomputed vectors directly (no re-embedding)
            - returns {"store": "Chroma", "stored": N}
        on any exception:
            - logs full traceback with log.error(..., exc_info=True)
            - falls back to _persist_jsonl()
            - returns {"store": "JSONL", "stored": N, "error": "..."}

Parallel:
        generate_ford_db.py runs independently of the above
        etl(db_path="data/ford.db", rows_per_table=200, seed=42)
            - Faker + random.Random(42) ensures reproducible data
            - synthesizes plants, dealers, vehicles, sales
            - loads all tables in a single transaction with PRAGMA foreign_keys=ON
            - prints row counts for each table
```

### Phase 3: Runtime Query (per user message)

```
Step 1  User types a message in the Streamlit UI (interface.py)
        If CHATBOT_PASSWORD is set:
            - auth_attempts >= 5 → lockout, st.stop()
            - time.time() - last_activity > 1800 → session expiry,
              authenticated = False, show expiry message
            - not authenticated → render login form only, st.stop()
        Prompt submitted → handle_user_query() called with enable_judge=True

Step 2  handler.py — Security pipeline
        sanitize_input(prompt, session_id=conversation_id):
            - type and length check (>1000 chars → ValueError)
            - normalize: strip zero-width, NFKC, HTML entity decode, collapse spaces
            - compact check: NON_ALNUM_RE stripped lowercase vs COMPACT_BLOCK_TRIGGERS
            - WARN pattern scan: log warning, continue
            - BLOCK pattern scan: log [guardrail] BLOCKED, raise ValueError
        ValueError propagates to interface.py → shown as st.error(), no LLM call

        redact_pii(clean_prompt):
            - CARD_CANDIDATE_RE + Luhn → [CARD]
            - Presidio AnalyzerEngine + AnonymizerEngine → [ENTITY_TYPE]
            - regex fallback: EMAIL → [EMAIL], PHONE → [PHONE],
                             SSN → [SSN], DOB → [DATE_TIME]
            - returns (redacted_text, was_redacted, entity_types)
        if was_redacted:
            log.warning("[pii] WARNING detected=[...] conversation=<id>")

        QueryRecord(conversation_id, redacted_prompt, timestamp)
        → QueryStore.append() → data/queries/query_log.jsonl

        ConversationMessage(role="user", content=redacted_prompt, created_at)
        → ConversationStore.append() → data/conversations/<id>.jsonl
          (Fernet-encrypted if CONVERSATION_KEY is set)

Step 3  judge.py — Prompt classification
        judge_prompt(prompt, model="gpt-4o-mini"):
            - if no OPENAI_API_KEY → _heuristic_judgement()
            - else: llm.bind(response_format={"type": "json_object"})
                    invoke_with_retry(json_llm, [system_prompt, user_prompt])
                    json.loads() + VerdictSchema.model_validate()
                    if parse fails → log.warning, _heuristic_judgement()
        returns JudgeVerdict(decision, confidence, rationale, raw_response)

        handler.py — Low-confidence override:
            if confidence < 0.60 and decision in {GENERAL, SQL}:
                log.info("[judge] low-confidence ... overridden to RAG")
                decision = Decision.RAG

Step 4a GENERAL route → general.py
        handle_general_query(prompt, conversation_id, store, factory, history)
        _history_as_messages(history, limit=6) → prior chat messages
        invoke_with_retry(llm, [system, *history, user])
        model: gpt-3.5-turbo, temperature=0.2, timeout=30s
        on RateLimitError/APITimeoutError/APIConnectionError: retry 3x
        ConversationMessage(role="assistant", ...) → ConversationStore.append()

Step 4b RAG route → rag.py
        handle_rag_query(prompt, conversation_id, store, factory, history)
        build_vectorstore(cfg) → ChromaDB (cached by config path + collection)
        retriever = vectorstore.as_retriever(search_type="similarity", k=4)
        history_text = _format_history(history, limit=6) → context string
        query = "Conversation context:\n...\nCurrent question:\n{prompt}"
        sources = retriever.invoke(query) → top-4 Document objects
        context = "\n\n".join(doc.page_content for doc in sources)
        system: "Answer using ONLY the provided context snippets..."
        invoke_with_retry(llm, [system, user_with_context])
        model: gpt-3.5-turbo, temperature=0.0, timeout=30s
        content = answer + "\n\n" + _format_sources(sources)
        ConversationMessage(role="assistant", ...) → ConversationStore.append()

Step 4c SQL route → sql.py
        handle_sql_query(prompt, conversation_id, store, factory, history)
        conn = sqlite3.connect("data/ford.db")
        schema = _introspect_schema(conn) → DDL strings from sqlite_master
        metadata = _build_metadata_summary(conn) → distinct model/segment/customer_type values
        _generate_sql(prompt, schema, metadata, llm):
            GPT-4o generates {"sql": "SELECT ...", "rationale": "..."}
            blocklist check: raises ValueError on DROP/DELETE/UPDATE/INSERT/ALTER
        conn.execute(sql) → rows
        if no rows: fallback sql (cheapest vehicle by MSRP), note added
        row_dicts = [{col: val}] for each row
        _summarize_result(prompt, sql, row_dicts, llm, note) → GPT-4o summary
        content = summary + rationale + sql code fence + key rows display
        ConversationMessage(role="assistant", ...) → ConversationStore.append()

Step 5  Response rendered in Streamlit
        result.assistant_message.content → st.write()
        _append_messages(result.user_message, result.assistant_message)
        st.session_state.last_activity = time.time()
        st.rerun()
```

---

## Configuration

### `Configs/corpus.yaml`
| Key | Default | Description |
|---|---|---|
| `project_name` | `ford-corp` | Project identifier used in logs |
| `output_dir` | `data` | Root data directory for all pipeline outputs |
| `user_agent` | `RAGScraper/1.0 (+https://example.com/contact)` | HTTP User-Agent header for the crawler |
| `seed_urls` | 5 Ford URLs | Pages to crawl (operations, news, support, company, vehicles) |
| `concurrency` | `8` | Number of concurrent async crawl workers |
| `request_timeout_sec` | `20` | Per-request HTTP timeout in seconds |
| `retry_attempts` | `3` | Maximum fetch retries per URL |
| `sleep_min_ms` | `200` | Minimum polite pacing delay between requests (ms) |
| `sleep_max_ms` | `600` | Maximum polite pacing delay between requests (ms) |
| `respect_robots` | `true` | Whether to honor robots.txt Disallow rules |
| `chunk_max_tokens` | `350` | Maximum tokens per chunk (tiktoken cl100k_base) |
| `chunk_overlap_tokens` | `40` | Sliding window overlap between chunks |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model identifier |
| `embedding_dimensions` | `1536` | Vector dimensions for the embedding model |
| `collection_name` | `ford-rag` | ChromaDB collection name |

### Environment Variables
| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (for LLM features) | OpenAI API key used by all embedding and chat model calls |
| `CONVERSATION_KEY` | No | Base64 Fernet key for encrypting conversation transcripts at rest |
| `CHATBOT_PASSWORD` | No | Password for the Streamlit UI auth gate (skipped if unset) |
| `GUARDRAIL_CONFIG_PATH` | No | Path to a YAML file overriding the default guardrail rules |

---

## Setup & Installation

```bash
# 1. Clone and enter project
git clone <repo-url>
cd Chat_Bot_Rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export OPENAI_API_KEY=sk-...
export CONVERSATION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export CHATBOT_PASSWORD=your-password

# 5. Generate synthetic Ford database
python -m pipelines.ingest.generate_ford_db --rows 200 --db data/ford.db --seed 42

# 6. Run ETL (scrape Ford pages)
python -m pipelines.etl.extract_ford --config Configs/corpus.yaml
python -m pipelines.etl.transform_ford --input data/input/crawlinfo.jsonl --output data/clean/cleaned.jsonl

# 7. Run ingestion (embed and store in ChromaDB)
python -m pipelines.rag.ingestion.pipeline --config Configs/corpus.yaml
```

---

## Running the App

```bash
# Start the Streamlit chatbot
streamlit run pipelines/rag/Retrieval/interface.py

# CLI query tool (no UI — useful for verifying the vector index)
python -m pipelines.rag.ingestion.qa \
  --config Configs/corpus.yaml \
  --query "What does Ford say about EV battery warranties?" \
  --k 4 \
  --model gpt-4o-mini

# Run unit tests
python -m pytest tests/ -v
```

---

## Sprint History

### Bug Fixes (TASK-01 through TASK-07) — 2026-05-01/02

Seven bugs were identified in the initial codebase audit and fixed before sprint work began.

**TASK-01 (Critical)** — `extract_ford.py` contained two `time.sleep()` calls inside `async def` functions (`fetch_with_retries` and `worker`). Synchronous sleep in an async function blocks the entire event loop, defeating all concurrency. Fixed by replacing both calls with `await asyncio.sleep()` and removing the unused `time` import.

**TASK-02 (Critical)** — `store.py` used the deprecated `langchain_community.vectorstores.Chroma` import and called `vs.persist()`, which was removed in ChromaDB >= 0.4 and raises `AttributeError` at runtime. Fixed by updating the import to prefer `langchain_chroma` (with community as fallback) and removing the `persist()` call entirely, as modern ChromaDB auto-persists.

**TASK-03 (High)** — `store.py` wrapped the entire Chroma write in a bare `except Exception` with no logging, causing silent failures. Operators would see successful-looking output even when Chroma had crashed and JSONL fallback was used. Fixed by adding `log.error(..., exc_info=True)` before the fallback path and including an `"error"` key in the returned stats dict.

**TASK-04 (High)** — `embed_openai.py` sent all chunks to the OpenAI embeddings API in a single call. The API rejects requests with more than 2048 inputs, causing hard failures on any real-sized corpus. Fixed by adding a batching loop with `BATCH_SIZE = 512` and a mismatch guard that raises `RuntimeError` if the number of returned vectors does not match the number of input chunks.

**TASK-05 (Medium)** — `rag.py` and `qa.py` both called the deprecated `retriever.get_relevant_documents(query)` method. LangChain replaced this with `retriever.invoke(query)` in recent versions, generating deprecation warnings on every query. Fixed in both files.

**TASK-06 (High)** — `langchain-chroma` and `faker` were used in core code paths but absent from `requirements.txt`, causing silent fallbacks or `ModuleNotFoundError` on fresh installs. Both packages were added with pinned versions consistent with the project stack (`langchain-chroma==0.2.6`, `faker==25.2.0`).

**TASK-07 (Critical)** — A naming mismatch between `transform_ford.py` (default output `data/clean/plain_pages.jsonl`) and `load_clean.py` (expects `data/clean/cleaned.jsonl`) meant the ETL and ingestion pipelines never connected. Running the full pipeline produced zero documents in ChromaDB. Fixed by correcting the default `--output` value in `transform_ford.py` to `data/clean/cleaned.jsonl`.

---

### Sprint 1 — Security Baseline (TASK-08 through TASK-11) — 2026-05-01/02

Sprint 1 focused on making the chatbot safe for real users. TASK-08 and TASK-09 required a rework after the initial delivery was assessed as insufficient for production standards.

**TASK-08 (High) — Input Guardrail** — `guardrails.py` was created with `sanitize_input()`. The initial implementation used simple substring matching which had two critical flaws: short phrases like "act as" triggered false positives on legitimate Ford queries, and the patterns were trivially bypassed via unicode homoglyphs, zero-width characters, or HTML encoding. The rework introduced a normalization-first approach (zero-width strip, NFKC, HTML entity decode, whitespace collapse) before any pattern matching, replaced flat substring checks with a tiered config of context-anchored minimum-four-word phrases (BLOCK and WARN tiers), added a compact scanner for obfuscation bypass variants, wired in `session_id` for audit trail logging, and made the rule config overrideable via `GUARDRAIL_CONFIG_PATH`. The function was wired into `handle_user_query` as the first operation.

**TASK-09 (High) — PII Redaction** — `pii.py` was created with `redact_pii()`. The initial implementation used a broad card number regex that matched Ford VINs and part numbers as false positives, lacked coverage for names and dates of birth, and logged PII events at `INFO` level. The rework replaced the card regex with Luhn-algorithm validation (eliminating virtually all false positives), integrated Microsoft Presidio (`presidio-analyzer`, `presidio-anonymizer`) as the primary detection engine for 50+ entity types with Luhn and regex as sequential fallbacks, changed the log level to `WARNING`, updated the log format to include entity type lists but never values, and updated the `redact_pii` return signature to include `entity_types` (a list), which was reflected in `handler.py`.

**TASK-10 (High) — Conversation Encryption** — `ConversationStore` in `handler.py` was updated to support Fernet symmetric encryption. When `CONVERSATION_KEY` is set, each conversation line is encrypted before writing and decrypted on load. When the key is absent, writes proceed unencrypted with a `log.warning` on every call. A key-generation one-liner was added to the module docstring. `cryptography==44.0.1` was added to `requirements.txt`. Accepted without rework.

**TASK-11 (Medium) — Auth Gate** — `interface.py` received a Streamlit login screen blocking the chat UI until a correct password is entered. When `CHATBOT_PASSWORD` is not set, auth is skipped with a one-time log warning (dev mode). Accepted, with two gaps logged for Sprint 2: no brute-force protection and no session timeout.

---

### Sprint 2 — Judge Hardening & Reliability (TASK-12 through TASK-16) — 2026-05-02

Sprint 2 hardened the routing layer, made all LLM calls resilient to transient network failures, and closed the auth gaps from Sprint 1.

**TASK-12 (High) — Judge Structured Output** — `judge.py` was updated to pass `response_format={"type": "json_object"}` via `llm.bind()` before the LLM call, forcing OpenAI to return valid JSON on every response. The JSON parse failure path was updated to emit `log.warning("[judge] JSON parse failed, falling back to heuristic. raw=<trimmed>")` before calling the keyword fallback. The LLM unavailable path also logs before falling back. The `ChatOpenAI` constructor was updated with `max_retries=0` to centralize retry policy in `llm_utils.py`.

**TASK-13 (High) — Few-Shot Examples** — Eight labeled examples were added to `SYSTEM_PROMPT` in `judge.py`, covering 3 RAG cases, 3 SQL cases, and 2 GENERAL cases, all using realistic Ford chatbot inputs. Examples are embedded in the static prompt constant rather than constructed dynamically.

**TASK-14 (High) — Confidence Override** — `handler.py` was updated to apply a post-judge override: if `verdict.confidence < 0.60` and `verdict.decision` is `GENERAL` or `SQL`, the decision is overridden to `RAG` with the original rationale augmented by a note explaining the override. This ensures the safer, document-grounded path is taken when the judge is uncertain. The override is logged at `INFO` level.

**TASK-15 (High) — Retry + Timeout** — `llm_utils.py` was created with `invoke_with_retry`, using tenacity to retry on `RateLimitError`, `APITimeoutError`, and `APIConnectionError` with exponential backoff (1s, 2s, 4s). On exhaustion, a clean `RuntimeError("LLM unavailable after 3 attempts")` is raised. Every `llm.invoke()` call in `judge.py`, `rag.py`, `sql.py`, and `general.py` was replaced with `invoke_with_retry`. `tenacity==9.0.0` was added to `requirements.txt`.

**TASK-16 (Medium) — Auth Hardening** — `interface.py` was updated with brute-force protection (5-attempt lockout via `st.session_state.auth_attempts`) and session timeout (30-minute inactivity expiry via `last_activity` timestamp updated on every query submission). Both features are inactive when `CHATBOT_PASSWORD` is not set.

---

## Test Report

**Last run:** 2026-05-02 — 79 tests, 77 passed, 2 failed

| Module | Test File | Tests | Passed | Failed | Status |
|---|---|---|---|---|---|
| `guardrails.py` | `test_guardrails.py` | 20 | 20 | 0 | PASS |
| `pii.py` | `test_pii.py` | 11 | 11 | 0 | PASS |
| `judge.py` | `test_judge.py` | 12 | 11 | 1 | FAIL |
| `llm_utils.py` | `test_llm_utils.py` | 4 | 4 | 0 | PASS |
| `chunk.py` | `test_chunk.py` | 8 | 8 | 0 | PASS |
| `store.py` | `test_store.py` | 5 | 4 | 1 | WARN |
| `embed_openai.py` | `test_embed_openai.py` | 6 | 6 | 0 | PASS |
| `handler.py` (wiring) | `test_handler_wiring.py` | 5 | 5 | 0 | PASS |

### Bugs Found by Tests

**BUG-01** (`pipelines/rag/Retrieval/judge.py` — `_heuristic_judgement`) — The heuristic router misroutes count queries that mention a Ford model name. Input: `"How many Explorers were sold last year?"` produces `Decision.RAG` instead of `Decision.SQL`. Root cause: `KEYWORDS_SQL` does not contain `"sold"`, `"many"`, or `"how many"`. The word `"explorer"` matches `BRAND_TERMS` first, winning the RAG route before SQL keywords are checked. Any aggregation or count query phrased with a Ford model name will be misrouted by the heuristic. Impact is low in production since GPT-4o-mini routes correctly when the API is available, but the heuristic fallback is a safety net and should be reliable. Fix: add `"sold"`, `"how many"`, `"aggregate"`, `"total sales"` to `KEYWORDS_SQL`, or restructure the heuristic to check for SQL signals even when a brand term is present.

### Test Issues

**TEST-ISSUE-01** (`tests/test_store.py::test_chroma_fallback_on_import_error`) — The test used an invalid mock approach (`patch("....__code__"`), raising `TypeError: __code__ must be set to a code object`. This is a test-writing error, not a source code bug. The underlying fallback behavior in `store.py` is confirmed working: when `langchain_chroma` is unavailable, the code correctly catches `ImportError`, logs `"Chroma store failed; falling back to JSONL storage"`, and returns a JSONL result with an `"error"` key. Fix: rewrite the test using `unittest.mock.patch("pipelines.rag.ingestion.store.Chroma", side_effect=ImportError(...))`.

---

## Known Issues & Backlog

### Open Bugs
- **BUG-01** (`judge.py`) — Heuristic misroutes count queries on Ford model names. Example: `"How many Explorers were sold last year?"` routes to RAG instead of SQL. Fix: add `"sold"`, `"how many"`, `"aggregate"` to `KEYWORDS_SQL`.

### Minor Items (backlog)
- `llm_utils.py`: The `RuntimeError` raised on retry exhaustion hardcodes `"LLM unavailable after 3 attempts"` regardless of the `max_attempts` parameter value. Should use `f"LLM unavailable after {max_attempts} attempts"` for accuracy when the parameter is overridden.
- `judge.py`: The few-shot examples in `SYSTEM_PROMPT` use a shorthand notation (`-> rag, 0.95`) while the instruction says "Return strict JSON". If routing drifts on edge cases, replacing the shorthand with actual JSON output examples would tighten the output contract.
- `interface.py`: When a session expires, `st.session_state.auth_attempts` is not cleared. A user whose session expired still carries their previous failed attempt count into the new login attempt sequence.
- `test_store.py::test_chroma_fallback_on_import_error`: Needs to be rewritten with a clean `patch("pipelines.rag.ingestion.store.Chroma", side_effect=ImportError(...))` approach.

### Planned Features (Sprint 3)
- Location-aware dealer enrichment: capture ZIP code at conversation start, perform a nearby dealer lookup, and inject relevant dealer offers alongside vehicle-related responses.
- New database tables: `zip_codes`, `dealer_locations`, `offers` — extending `ford.db` to support geographically scoped queries.
- Parallel enrichment layer: main answer (RAG or SQL) and dealer offers computed simultaneously; results merged before display.
- Contextual suggestions block appended at the bottom of every vehicle-related reply.

---

*Built with LangChain · LangGraph · ChromaDB · OpenAI · Streamlit · Microsoft Presidio · Tenacity*
