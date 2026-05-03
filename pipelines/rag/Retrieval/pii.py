"""PII detection and redaction helpers for Retrieval flows."""

from __future__ import annotations

import logging
import re

try:  # pragma: no cover - optional dependency at runtime
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except Exception:  # pragma: no cover
    AnalyzerEngine = None  # type: ignore[assignment]
    AnonymizerEngine = None  # type: ignore[assignment]
    OperatorConfig = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DOB_RE = re.compile(
    r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"
)
CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

_analyzer_engine = None
_anonymizer_engine = None
_presidio_init_attempted = False
_presidio_warning_logged = False


def _luhn_is_valid(number: str) -> bool:
    if not number.isdigit():
        return False
    checksum = 0
    parity = len(number) % 2
    for idx, char in enumerate(number):
        digit = int(char)
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _redact_luhn_cards(text: str) -> tuple[str, set[str]]:
    detected: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        digits = re.sub(r"[^0-9]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_is_valid(digits):
            detected.add("CREDIT_CARD")
            return "[CARD]"
        return match.group(0)

    return CARD_CANDIDATE_RE.sub(_replace, text), detected


def _regex_fallback_redaction(text: str) -> tuple[str, set[str]]:
    detected: set[str] = set()
    redacted = text
    patterns = [
        (EMAIL_RE, "[EMAIL]", "EMAIL_ADDRESS"),
        (PHONE_RE, "[PHONE]", "PHONE_NUMBER"),
        (SSN_RE, "[SSN]", "US_SSN"),
        (DOB_RE, "[DATE_TIME]", "DATE_TIME"),
    ]
    for pattern, replacement, entity_type in patterns:
        redacted, count = pattern.subn(replacement, redacted)
        if count > 0:
            detected.add(entity_type)
    return redacted, detected


def _get_presidio_engines():
    global _analyzer_engine, _anonymizer_engine, _presidio_init_attempted, _presidio_warning_logged

    if _analyzer_engine is not None and _anonymizer_engine is not None:
        return _analyzer_engine, _anonymizer_engine
    if _presidio_init_attempted:
        return None, None

    _presidio_init_attempted = True
    if AnalyzerEngine is None or AnonymizerEngine is None:
        if not _presidio_warning_logged:
            log.warning(
                "Presidio packages unavailable; using regex-only PII redaction fallback"
            )
            _presidio_warning_logged = True
        return None, None

    try:
        _analyzer_engine = AnalyzerEngine()
        _anonymizer_engine = AnonymizerEngine()
        return _analyzer_engine, _anonymizer_engine
    except Exception as exc:
        if not _presidio_warning_logged:
            log.warning(
                "Presidio initialization failed; using regex-only PII redaction fallback (%s)",
                exc,
            )
            _presidio_warning_logged = True
        return None, None


def _presidio_redaction(text: str) -> tuple[str, set[str]]:
    global _presidio_warning_logged
    analyzer, anonymizer = _get_presidio_engines()
    if analyzer is None or anonymizer is None or OperatorConfig is None:
        return text, set()

    try:
        results = analyzer.analyze(text=text, language="en")
        if not results:
            return text, set()
        detected_types = {result.entity_type for result in results}
        operators = {
            entity_type: OperatorConfig("replace", {"new_value": f"[{entity_type}]"})
            for entity_type in detected_types
        }
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        return anonymized.text, detected_types
    except Exception as exc:
        if not _presidio_warning_logged:
            log.warning(
                "Presidio redaction failed; using regex-only PII redaction fallback (%s)",
                exc,
            )
            _presidio_warning_logged = True
        return text, set()


def redact_pii(text: str) -> tuple[str, bool, list[str]]:
    """Redact PII and return text, redaction flag, and detected entity types."""

    redacted = text or ""
    detected_types: set[str] = set()

    redacted, card_types = _redact_luhn_cards(redacted)
    detected_types.update(card_types)

    presidio_redacted, presidio_types = _presidio_redaction(redacted)
    if presidio_types:
        redacted = presidio_redacted
        detected_types.update(presidio_types)

    fallback_redacted, fallback_types = _regex_fallback_redaction(redacted)
    redacted = fallback_redacted
    detected_types.update(fallback_types)

    sorted_types = sorted(detected_types)
    return redacted, bool(sorted_types), sorted_types


__all__ = ["redact_pii"]
