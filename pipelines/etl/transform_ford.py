# pipelines/etl/transform_ford.py
"""
HTML → Readable Plain Text (NO CHUNKING)

Reads:  data/input/crawlinfo.jsonl
Writes: data/clean/plain_pages.jsonl
"""

import argparse
import hashlib
import html as html_lib
import json
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


# ---- I/O ----
def ensure_new_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return path

def read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[transform] skip malformed JSON at line {ln}: {e}")

def append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


# ---- HTML → text ----
def html_to_text_bs4(html: str) -> Tuple[str, str]:
    parser = "lxml" if BeautifulSoup and "lxml" else "html.parser"
    soup = BeautifulSoup(html, parser)
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text

def html_to_text_naive(html: str) -> Tuple[str, str]:
    body = re.sub(r"(?is)<script.*?>.*?</script>", "", html)
    body = re.sub(r"(?is)<style.*?>.*?</style>", "", body)
    m = re.search(r"(?is)<title>(.*?)</title>", html)
    title = m.group(1).strip() if m else ""
    body = re.sub(r"(?is)<[^>]+>", " ", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\s*\n\s*", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return title, body.strip()

def html_to_text(html: str) -> Tuple[str, str]:
    return html_to_text_bs4(html) if BeautifulSoup else html_to_text_naive(html)


# ---- Normalization (no chunking) ----
def unescape_html_entities(text: str) -> str:
    return html_lib.unescape(text).replace("\xa0", " ")

def normalize_bullets_and_dashes(text: str) -> str:
    text = re.sub(r"^[\t >]*[•·]\s*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^[\t >]*[-–—]\s*", "- ", text, flags=re.MULTILINE)
    return text.replace("–", "-").replace("—", "-")

def collapse_blank_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text)

def collapse_spaces_not_across_newlines(text: str) -> str:
    return re.sub(r"[^\S\r\n]{2,}", " ", text)

def unwrap_hard_wrapped_sentences(text: str) -> str:
    def _join(m): return f"{m.group(1)} {m.group(2)}"
    pattern = re.compile(r"([a-z0-9,\)\]])\n(?!\s*(?:- |\* |#|[0-9]+\.\s))([a-z])", re.MULTILINE)
    return pattern.sub(_join, text)

def trim_lines_and_document(text: str) -> str:
    return "\n".join(ln.strip() for ln in text.split("\n")).strip()

def normalize_text_pipeline(text: str) -> str:
    text = unescape_html_entities(text)
    text = normalize_bullets_and_dashes(text)
    text = collapse_blank_lines(text)
    text = collapse_spaces_not_across_newlines(text)
    text = unwrap_hard_wrapped_sentences(text)
    return trim_lines_and_document(text)


# ---- Records ----
def stable_id_from_url(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()

def build_page_record(url: str, fetched_at: Optional[str], title: str, text: str) -> Dict:
    return {
        "doc_id": stable_id_from_url(url),
        "url": url,
        "title": title,
        "fetched_at": fetched_at,
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
    }


# ---- Core ----
def transform_to_plain(input_jsonl: Path, output_jsonl: Path) -> None:
    ensure_new_file(output_jsonl)
    total_in = total_written = 0

    for rec in read_jsonl(input_jsonl):
        total_in += 1
        status = rec.get("status")
        html = rec.get("html")
        ctype = (rec.get("content_type") or "").lower()
        if status != 200 or html is None or ("text/" not in ctype and "html" not in ctype):
            continue

        title, raw_text = html_to_text(html)
        clean_text = normalize_text_pipeline(raw_text)

        out = build_page_record(rec.get("url"), rec.get("fetched_at"), title, clean_text)
        append_jsonl(output_jsonl, out)
        total_written += 1

    print(f"[transform] pages_seen={total_in} pages_written={total_written} → {output_jsonl}")


# ---- CLI ----
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HTML → readable plain text (no chunking)")
    ap.add_argument("--input", default="data/input/crawlinfo.jsonl")
    ap.add_argument("--output", default="data/clean/plain_pages.jsonl")
    return ap.parse_args()

def main():
    args = parse_args()
    transform_to_plain(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()
