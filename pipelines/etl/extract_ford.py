"""
EXTRACT PHASE (Ford)
Fetch seed URLs from corpus.yaml and write raw results to:
  data/input/crawlinfo.jsonl  (one JSON object per line)
No HTML parsing/cleaning here—only network I/O + bookkeeping.
"""

import argparse
import asyncio
import json
import random
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml
import logging

# ---------- Logging ----------
LOG_LEVEL = logging.getLevelName(
    (Path(".env").read_text().split("LOG_LEVEL=")[1].splitlines()[0].strip()
     if Path(".env").exists() and "LOG_LEVEL=" in Path(".env").read_text()
     else "INFO")
) if isinstance(logging.getLevelName("INFO"), int) else logging.INFO

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("extract_ford")

# ---------- Config ----------
@dataclass
class CrawlConfig:
    project_name: str
    output_dir: str
    user_agent: str
    concurrency: int
    request_timeout_sec: int
    retry_attempts: int
    sleep_min_ms: int
    sleep_max_ms: int
    respect_robots: bool
    seed_urls: List[str]
    exclude_url_patterns: List[str]

def load_config(path: str) -> CrawlConfig:
    """Load crawl settings from YAML into a typed config."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = CrawlConfig(
        project_name=raw.get("project_name", "ford-knowledge"),
        output_dir=raw.get("output_dir", "data"),
        user_agent=raw.get("user_agent", "RAGScraper/1.0 (+https://example.com/contact)"),
        concurrency=int(raw.get("concurrency", 6)),
        request_timeout_sec=int(raw.get("request_timeout_sec", 20)),
        retry_attempts=int(raw.get("retry_attempts", 3)),
        sleep_min_ms=int(raw.get("sleep_min_ms", 300)),
        sleep_max_ms=int(raw.get("sleep_max_ms", 900)),
        respect_robots=bool(raw.get("respect_robots", True)),
        seed_urls=list(raw.get("seed_urls", [])),
        exclude_url_patterns=list(raw.get("exclude_url_patterns", [])),
    )
    log.debug("Loaded config: %r", cfg)
    return cfg

# ---------- Filesystem ----------
def ensure_output_file(base_dir: Path) -> Path:
    """Create data/input/ and return path to fresh crawlinfo.jsonl."""
    input_dir = base_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_path = input_dir / "crawlinfo.jsonl"
    if out_path.exists():
        log.info("Output exists, removing: %s", out_path)
        out_path.unlink()
    log.info("Writing to: %s", out_path)
    return out_path

def append_jsonl(path: Path, obj: dict) -> None:
    """Append one JSON object as a single line."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")

# ---------- robots.txt ----------
@dataclass
class RobotsRules:
    disallows: List[str]  # active Disallow rules under User-agent: *

async def fetch_text(client: httpx.AsyncClient, url: str, timeout: int) -> Optional[str]:
    """GET url and return text on 200, else None."""
    try:
        resp = await client.get(url, timeout=timeout, follow_redirects=True)
        return resp.text if resp.status_code == 200 else None
    except Exception as e:
        log.debug("fetch_text error for %s: %s", url, e)
        return None

async def fetch_robots(client: httpx.AsyncClient, any_url: str, timeout: int) -> RobotsRules:
    """Fetch and parse robots.txt for the host of any_url (User-agent: * only)."""
    try:
        parts = urllib.parse.urlsplit(any_url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        text = await fetch_text(client, robots_url, timeout)

        disallows: List[str] = []
        if text:
            user_agent_block_applies = False
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                lower = line.lower()
                if lower.startswith("user-agent:"):
                    ua_value = line.split(":", 1)[1].strip()
                    user_agent_block_applies = (ua_value == "*" or ua_value == '"*"')
                elif user_agent_block_applies and lower.startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    disallows.append(path or "/")  # empty path → treat as "/"
        log.debug("robots for %s → %s", any_url, disallows)
        return RobotsRules(disallows=disallows)
    except Exception as e:
        log.debug("fetch_robots failed for %s: %s", any_url, e)
        return RobotsRules(disallows=[])

def is_allowed_by_robots(path: str, rules: RobotsRules) -> bool:
    """Return False if path starts with any disallowed prefix (or full-site '/')."""
    for rule in rules.disallows:
        if rule == "/" or (rule and path.startswith(rule)):
            return False
    return True

# ---------- Networking ----------
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
}

async def fetch_with_retries(
    client: httpx.AsyncClient,
    url: str,
    timeout: int,
    attempts: int
):
    """GET with limited retries + jittered backoff. Return (status, ctype, body_or_none)."""
    for i in range(attempts):
        try:
            log.debug("GET %s (attempt %d/%d)", url, i + 1, attempts)
            r = await client.get(url, timeout=timeout, follow_redirects=True)
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and ("text/html" in ctype or "text/" in ctype):
                return r.status_code, ctype, r.text
            return r.status_code, ctype, None
        except Exception as e:
            sleep_s = min(1.5 ** i, 8.0) + random.random() * 0.3
            log.debug("Error '%s' on %s → backoff %.2fs", e, url, sleep_s)
            time.sleep(sleep_s)
    return None, None, None

# ---------- Orchestrator ----------
async def extract_once(cfg: CrawlConfig):
    """
    Orchestrate extract:
      - prepare output
      - per-host robots cache
      - filter seeds
      - concurrent workers → fetch → append JSONL
    """
    out_path = ensure_output_file(Path(cfg.output_dir))

    headers = {"User-Agent": cfg.user_agent, **DEFAULT_HEADERS}
    async with httpx.AsyncClient(headers=headers) as client:
        robots_cache: Dict[str, RobotsRules] = {}

        async def allowed(u: str) -> bool:
            """Check robots.txt for URL u (cached per host)."""
            if not cfg.respect_robots:
                return True
            parts = urllib.parse.urlsplit(u)
            host = f"{parts.scheme}://{parts.netloc}"
            if host not in robots_cache:
                robots_cache[host] = await fetch_robots(client, u, cfg.request_timeout_sec)
            return is_allowed_by_robots(parts.path or "/", robots_cache[host])

        # Optional exclude patterns (e.g., skip heavy assets if configured)
        seeds: List[str] = []
        for u in cfg.seed_urls:
            if any(pat and pat in u for pat in (cfg.exclude_url_patterns or [])):
                log.info("Excluding by pattern: %s", u)
                continue
            seeds.append(u)

        sem = asyncio.Semaphore(cfg.concurrency)

        async def worker(u: str):
            """Fetch one URL and append a single JSON record to crawlinfo.jsonl."""
            async with sem:
                # robots gate
                try:
                    if not await allowed(u):
                        log.warning("[robots] Disallowed: %s", u)
                        append_jsonl(out_path, {
                            "url": u,
                            "status": 999,
                            "content_type": None,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "html": None,
                            "note": "blocked_by_robots"
                        })
                        return
                except Exception as e:
                    log.warning("[robots] Check failed for %s (%s). Skipping.", u, e)
                    append_jsonl(out_path, {
                        "url": u,
                        "status": 998,
                        "content_type": None,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "html": None,
                        "note": "robots_check_failed"
                    })
                    return

                # fetch
                status, ctype, body = await fetch_with_retries(
                    client, u, cfg.request_timeout_sec, cfg.retry_attempts
                )

                # record outcome
                append_jsonl(out_path, {
                    "url": u,
                    "status": status,
                    "content_type": ctype,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "html": body
                })

                # polite pacing
                time.sleep(random.randint(cfg.sleep_min_ms, cfg.sleep_max_ms) / 1000.0)

        await asyncio.gather(*(worker(u) for u in seeds))

    log.info("Extract complete → %s", out_path)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="corpus.yaml", help="Path to corpus.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info("Project: %s | Seeds: %d | Concurrency: %d",
             cfg.project_name, len(cfg.seed_urls), cfg.concurrency)

    asyncio.run(extract_once(cfg))

if __name__ == "__main__":
    main()
