"""Input guardrails for retrieval flows."""

from __future__ import annotations

import html
import logging
import os
import re
import unicodedata
from pathlib import Path

try:  # pragma: no cover - optional dependency at runtime
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

MAX_PROMPT_CHARS = 1000
INJECTION_ERROR = "Prompt rejected: potential injection detected"

log = logging.getLogger(__name__)

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Tiered config (BLOCK/WARN). Can be overridden with GUARDRAIL_CONFIG_PATH.
DEFAULT_GUARDRAIL_CONFIG: dict[str, list[dict[str, str]]] = {
    "BLOCK": [
        {
            "name": "ignore_previous_instructions",
            "pattern": r"\bignore\s+(?:all\s+|any\s+|the\s+)?previous\s+instructions\b",
        },
        {
            "name": "ignore_all_instructions",
            "pattern": r"\bignore\s+all\s+instructions\b",
        },
        {
            "name": "system_role_override",
            "pattern": r"(?:^|\n)\s*(?:system\s*:|<\s*system\s*>|\[\s*system\s*])",
        },
        {
            "name": "you_are_now_unrestricted",
            "pattern": r"\byou\s+are\s+now\s+(?:in\s+)?(?:developer|god|jailbreak|unrestricted)\s+mode\b",
        },
        {
            "name": "act_as_no_restrictions",
            "pattern": r"\bact\s+as\s+if\s+you\s+have\s+no\s+restrictions\b",
        },
        {
            "name": "pretend_unfiltered_assistant",
            "pattern": r"\bpretend\s+you\s+are\s+(?:an?\s+)?(?:unfiltered|unrestricted)\s+assistant\b",
        },
        {
            "name": "disregard_prior_context",
            "pattern": r"\bdisregard\s+all\s+(?:prior|previous)\s+(?:instructions|context)\b",
        },
        {
            "name": "forget_safety_rules",
            "pattern": r"\bforget\s+your\s+(?:safety|policy|rules|instructions|constraints)\b",
        },
    ],
    "WARN": [
        {
            "name": "jailbreak_keyword",
            "pattern": r"\b(?:jailbreak|disable\s+guardrails|bypass\s+safety)\b",
        },
        {
            "name": "policy_override_hint",
            "pattern": r"\b(?:override\s+the\s+policy|ignore\s+the\s+policy)\b",
        },
    ],
}

_compiled_rules: dict[str, list[tuple[str, re.Pattern[str]]]] | None = None

COMPACT_BLOCK_TRIGGERS: dict[str, str] = {
    "ignorepreviousinstructions": "ignore_previous_instructions",
    "ignoreallinstructions": "ignore_all_instructions",
    "disregardallpriorcontext": "disregard_prior_context",
    "disregardallpreviouscontext": "disregard_prior_context",
    "forgetyourinstructions": "forget_safety_rules",
    "forgetyourrules": "forget_safety_rules",
}


def _normalize_for_scanning(prompt: str) -> str:
    normalized = ZERO_WIDTH_RE.sub("", prompt)
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = html.unescape(normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _validate_config(config: object) -> bool:
    if not isinstance(config, dict):
        return False
    for tier in ("BLOCK", "WARN"):
        rules = config.get(tier)
        if not isinstance(rules, list):
            return False
        for rule in rules:
            if not isinstance(rule, dict):
                return False
            if not isinstance(rule.get("name"), str):
                return False
            if not isinstance(rule.get("pattern"), str):
                return False
    return True


def _load_guardrail_config() -> dict[str, list[dict[str, str]]]:
    config_path = os.getenv("GUARDRAIL_CONFIG_PATH")
    if not config_path:
        return DEFAULT_GUARDRAIL_CONFIG
    if yaml is None:
        log.warning(
            "[guardrail] PyYAML unavailable; cannot load config path=%s, using defaults",
            config_path,
        )
        return DEFAULT_GUARDRAIL_CONFIG

    try:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(
            "[guardrail] Failed to read config path=%s; using defaults (%s)",
            config_path,
            exc,
        )
        return DEFAULT_GUARDRAIL_CONFIG

    if _validate_config(raw):
        return raw  # type: ignore[return-value]

    log.warning(
        "[guardrail] Invalid config format at path=%s; using defaults",
        config_path,
    )
    return DEFAULT_GUARDRAIL_CONFIG


def _get_compiled_rules() -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    global _compiled_rules
    if _compiled_rules is None:
        compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        config = _load_guardrail_config()
        for tier in ("BLOCK", "WARN"):
            compiled[tier] = [
                (rule["name"], re.compile(rule["pattern"], re.IGNORECASE))
                for rule in config[tier]
            ]
        _compiled_rules = compiled
    return _compiled_rules


def sanitize_input(prompt: str, session_id: str | None = None) -> str:
    """Validate prompt content and reject likely prompt-injection payloads."""

    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string")

    stripped = prompt.strip()
    if not stripped:
        raise ValueError("Prompt must not be empty")
    if len(stripped) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds maximum length of {MAX_PROMPT_CHARS} characters")

    normalized = _normalize_for_scanning(stripped)
    if len(normalized) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds maximum length of {MAX_PROMPT_CHARS} characters")

    session = session_id or "unknown"
    rules = _get_compiled_rules()
    compact = NON_ALNUM_RE.sub("", normalized.lower())

    for compact_pattern, rule_name in COMPACT_BLOCK_TRIGGERS.items():
        if compact_pattern in compact:
            log.warning(
                "[guardrail] BLOCKED reason=%s session=%s length=%d",
                rule_name,
                session,
                len(normalized),
            )
            raise ValueError(INJECTION_ERROR)

    for rule_name, pattern in rules["WARN"]:
        if pattern.search(normalized):
            log.warning(
                "[guardrail] WARN reason=%s session=%s length=%d",
                rule_name,
                session,
                len(normalized),
            )

    for rule_name, pattern in rules["BLOCK"]:
        if pattern.search(normalized):
            log.warning(
                "[guardrail] BLOCKED reason=%s session=%s length=%d",
                rule_name,
                session,
                len(normalized),
            )
            raise ValueError(INJECTION_ERROR)

    return stripped


__all__ = ["sanitize_input", "MAX_PROMPT_CHARS", "INJECTION_ERROR"]
