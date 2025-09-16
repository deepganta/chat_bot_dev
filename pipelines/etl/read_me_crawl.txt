his document explains the Extract step of our ETL pipeline in detail: what each function does, how they connect, and the full execution flow. The goal of this phase is only to fetch pages and persist exactly what we received into a single JSONL file:

data/input/crawlinfo.jsonl


Each line is a JSON record:

{
  "url": "<seed url>",
  "status": 200,
  "content_type": "text/html; charset=utf-8",
  "fetched_at": "2025-09-15T19:30:01Z",
  "html": "<raw html or null>"
}

Quick Start
# run from project root
python -m pipelines.etl.extract_ford --config corpus.yaml
# output: data/input/crawlinfo.jsonl (overwritten each run)


corpus.yaml drives the behavior (seeds, headers, concurrency, sleep, retries, robots, etc.).

File: pipelines/etl/extract_ford.py
Module Purpose

Read seeds and crawl settings from corpus.yaml

Respect robots.txt (minimal but polite)

Fetch each seed URL (with retries/backoff and concurrency control)

Append one JSON object per URL to data/input/crawlinfo.jsonl

No cleaning/regex/chunking here (that’s the Transform phase)

Functions & What They Do
1) Logging Setup

Configures logging with level and format so you can see what is happening and why.

Tip: set LOG_LEVEL=DEBUG in .env for deep traces.

2) @dataclass CrawlConfig

Holds all runtime knobs loaded from YAML:

project_name, output_dir, user_agent

concurrency, request_timeout_sec, retry_attempts

sleep_min_ms, sleep_max_ms (polite delay jitter)

respect_robots (enable/disable robots checks)

seed_urls (list of starting URLs)

exclude_url_patterns (optional “skip” substrings)

Why: makes behavior data-driven and explicit.

3) load_config(path: str) -> CrawlConfig

Reads corpus.yaml → returns a fully-populated CrawlConfig.

Applies sensible defaults if fields are absent.

Debug-logs the result so you can confirm the settings used.

Trigger: called by main() before extraction begins.

4) ensure_output_file(base_dir: Path) -> Path

Ensures data/input/ exists.

Idempotent: deletes old crawlinfo.jsonl so each run is clean.

Returns the absolute Path to data/input/crawlinfo.jsonl.

Trigger: first thing inside extract_once().

5) append_jsonl(path: Path, obj: dict) -> None

Appends a single JSON object to the output file (one line per record).

Keeps memory use tiny; plays well with streaming and debugging.

Trigger: used by the worker() for every URL outcome (allowed/blocked/failed).

6) @dataclass RobotsRules

Tiny container: disallows: List[str]

Represents the active Disallow rules for User-agent: * on a host.

7) fetch_text(client, url, timeout) -> Optional[str] (async)

Generic “GET text” helper, returns response .text on HTTP 200 or None otherwise.

Used for fetching robots.txt.

Trigger: called by fetch_robots().

8) fetch_robots(client, any_url, timeout) -> RobotsRules (async)

Purpose: download and parse robots.txt for the host of any_url.

Steps:

Parse any_url into (scheme, netloc, path) and build robots.txt URL.

fetch_text() to retrieve file content.

Parse line-by-line:

Track whether we’re inside a User-agent: * block.

For each Disallow: under that block, collect its path (empty path → treated as /).

Return RobotsRules(disallows=[...]). On error, return an empty ruleset.

Why: polite crawling—skip paths the site asks us not to fetch.

Trigger: used inside the inner allowed(u) function (within extract_once) and cached per host.

9) is_allowed_by_robots(path: str, rules: RobotsRules) -> bool

Returns False if:

a rule is / (site-wide block), or

path.startswith(rule) for any disallowed rule.

Otherwise returns True.

Trigger: used by allowed(u).

10) fetch_with_retries(client, url, timeout, attempts) (async)

Resilient network fetch:

Try up to attempts times (e.g., 3).

On success (status 200 and content-type includes text/html or text/), return (status, ctype, html).

On non-HTML or non-200: return (status, ctype, None) (transparency for debugging).

On exceptions: jittered backoff, retry.

After exhausting attempts: return (None, None, None).

Trigger: called by worker(u) when URL is allowed.

11) extract_once(cfg: CrawlConfig) (async)

The orchestrator of the Extract step.

Key pieces inside:

a) Prepare output & HTTP client

out_path = ensure_output_file(Path(cfg.output_dir))

Create one httpx.AsyncClient with headers:

User-Agent from config

Accept: text/html,application/xhtml+xml;q=0.9,*/*;q=0.1

b) Robots cache & allowed(u) helper

robots_cache: Dict[str, RobotsRules] = {}

allowed(u):

If cfg.respect_robots is False, return True.

Parse host from u. If host not in cache, fetch_robots() and store.

Use is_allowed_by_robots(path, rules) → True/False.

c) Pre-filter seeds

Start from cfg.seed_urls.

Optionally skip URLs matching any exclude_url_patterns substrings.

d) Concurrency control

sem = asyncio.Semaphore(cfg.concurrency) → limit in-flight workers.

e) worker(u) (async)

Handles one URL:

Robots gate

if not await allowed(u):
    append_jsonl(... {"status": 999, "note": "blocked_by_robots"})
    return


Fetch with retries

status, ctype, body = await fetch_with_retries(...)


Record exactly what happened

Append one JSON object with url, status, content_type, fetched_at, html.

Polite sleep (random between sleep_min_ms and sleep_max_ms).

f) Launch workers

await asyncio.gather(*(worker(u) for u in seeds))

Result: one JSONL line per seed URL, documenting exactly what happened.

12) main()

CLI wrapper: parse --config, load_config(), then asyncio.run(extract_once(cfg)).

Call Flow — Who triggers whom?

Below is a step-by-step flow of a single run:

main()
 └─ load_config("corpus.yaml")  ➜ CrawlConfig
 └─ asyncio.run(extract_once(cfg))
     ├─ ensure_output_file(Path(cfg.output_dir))  ➜ data/input/crawlinfo.jsonl
     ├─ create httpx.AsyncClient(headers=UA + Accept)
     ├─ robots_cache = {}
     ├─ define allowed(u):
     │    ├─ if cfg.respect_robots == False → return True
     │    ├─ parts = urllib.parse.urlsplit(u)
     │    ├─ host = f"{parts.scheme}://{parts.netloc}"
     │    ├─ if host not in robots_cache:
     │    │    └─ robots_cache[host] = fetch_robots(client, u, cfg.timeout)
     │    └─ return is_allowed_by_robots(parts.path, robots_cache[host])
     ├─ seeds = [u for u in cfg.seed_urls if not excluded]
     ├─ sem = asyncio.Semaphore(cfg.concurrency)
     ├─ define worker(u):
     │    ├─ if not await allowed(u):
     │    │    └─ append_jsonl(..., {"status": 999, "note": "blocked_by_robots"}); return
     │    ├─ (status, ctype, body) = await fetch_with_retries(client, u, timeout, attempts)
     │    ├─ append_jsonl(..., {"url": u, "status": status, "content_type": ctype,
     │    │                     "fetched_at": <UTC ISO>, "html": body})
     │    └─ polite sleep (random between cfg.sleep_min_ms..max)
     └─ asyncio.gather(worker(u) for u in seeds)
         └─ completes → file ready