"""Runtime health checks for Streamlit sidebar status."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

try:  # pragma: no cover - optional dependency at runtime
    from cryptography.fernet import Fernet
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore[assignment]


def _safe_bool(check_fn) -> bool:
    try:
        return bool(check_fn())
    except Exception:
        return False


def _openai_key_present() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _conversation_key_enabled() -> bool:
    key = os.getenv("CONVERSATION_KEY", "").strip()
    if not key or Fernet is None:
        return False
    Fernet(key)
    return True


def _sqlite_readable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.execute("SELECT 1")
    return True


def _guardrail_mode() -> str:
    try:
        config_path = os.getenv("GUARDRAIL_CONFIG_PATH", "").strip()
        if not config_path:
            return "default"
        return "custom" if Path(config_path).exists() else "default"
    except Exception:
        return "default"


def check_health() -> dict:
    """Return non-raising runtime health status for key dependencies."""

    return {
        "openai_key": _safe_bool(_openai_key_present),
        "conversation_key": _safe_bool(_conversation_key_enabled),
        "ford_db": _safe_bool(lambda: _sqlite_readable(Path("data/ford.db"))),
        "chroma_index": _safe_bool(
            lambda: _sqlite_readable(Path("data/index/chroma.sqlite3"))
        ),
        "guardrail_config": _guardrail_mode(),
    }


__all__ = ["check_health"]
