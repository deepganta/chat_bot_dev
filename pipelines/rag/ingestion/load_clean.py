# ingestion/load_clean.py
from typing import List, Dict, Iterator
import os, json, glob

def _iter_jsonl(path: str) -> Iterator[Dict]:
    """Yield one dict per line from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():               # skip blank lines
                yield json.loads(line)

def load_cleaned(output_dir: str = "data") -> List[Dict]:
    """
    Load already-cleaned documents produced by your existing pipeline.

    Supports two formats:
      1) data/clean/cleaned.jsonl  -> rows like {"url": "...", "text": "...", "meta": {...}}
      2) data/clean/*.txt          -> plain text files; we synthesize minimal metadata
    """
    clean_dir = os.path.join(output_dir, "clean")
    records: List[Dict] = []

    jsonl_path = os.path.join(clean_dir, "cleaned.jsonl")
    if os.path.exists(jsonl_path):
        # Preferred path: JSONL keeps url + text + metadata together.
        for r in _iter_jsonl(jsonl_path):
            records.append({
                "url": r.get("url", ""),   # source url (if available)
                "text": r.get("text", ""), # cleaned text
                "meta": r.get("meta", {}), # arbitrary metadata dict
            })
        return records

    # Fallback: gather plain .txt files if JSONL isn't present.
    for path in glob.glob(os.path.join(clean_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # If the filename encodes the URL, you can decode it here.
        # Otherwise we keep url blank and stash the filename in meta.
        records.append({
            "url": "",
            "text": text,
            "meta": {"source_file": os.path.basename(path)}
        })

    return records
