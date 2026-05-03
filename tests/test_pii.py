"""Unit tests for pipelines/rag/Retrieval/pii.py"""
import pytest
from pipelines.rag.Retrieval.pii import redact_pii


# ── Return type contract ────────────────────────────────────────────────────

def test_return_type_clean_text():
    text, flag, types = redact_pii("I want a Ford Mustang")
    assert isinstance(text, str)
    assert isinstance(flag, bool)
    assert isinstance(types, list)

def test_clean_text_unchanged():
    text, flag, types = redact_pii("I want a Ford Mustang")
    assert text == "I want a Ford Mustang"
    assert flag is False
    assert types == []


# ── Email ────────────────────────────────────────────────────────────────────

def test_email_redacted():
    text, flag, types = redact_pii("Contact me at john.doe@example.com please")
    assert "[EMAIL]" in text
    assert "john.doe@example.com" not in text
    assert flag is True

def test_email_entity_type_logged():
    _, _, types = redact_pii("email: test@gmail.com")
    assert any("EMAIL" in t.upper() for t in types)


# ── Phone ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phone", [
    "214-555-1234",
    "(214) 555-1234",
    "2145551234",
    "+1 214 555 1234",
])
def test_phone_redacted(phone):
    text, flag, _ = redact_pii(f"Call me at {phone}")
    assert "[PHONE]" in text
    assert flag is True


# ── SSN ──────────────────────────────────────────────────────────────────────

def test_ssn_redacted():
    text, flag, types = redact_pii("My SSN is 123-45-6789")
    assert "[SSN]" in text
    assert "123-45-6789" not in text
    assert flag is True


# ── Credit card (Luhn) ───────────────────────────────────────────────────────

def test_valid_luhn_card_redacted():
    # 4111111111111111 is the canonical Luhn-valid Visa test number
    text, flag, _ = redact_pii("My card is 4111111111111111")
    assert "[CARD]" in text
    assert "4111111111111111" not in text
    assert flag is True

def test_invalid_luhn_not_redacted_as_card():
    # 1234567890123456 fails Luhn
    text, flag, _ = redact_pii("Number: 1234567890123456")
    assert "[CARD]" not in text

def test_vin_not_redacted_as_card():
    # VINs contain letters which break the digit run
    text, flag, _ = redact_pii("VIN: 1FTEW1E53KFC12345")
    assert "[CARD]" not in text


# ── Multiple PII in one string ───────────────────────────────────────────────

def test_multiple_pii_types_all_redacted():
    text, flag, types = redact_pii("Email: user@test.com, SSN: 123-45-6789")
    assert "[EMAIL]" in text or "user@test.com" not in text
    assert "[SSN]" in text or "123-45-6789" not in text
    assert flag is True
    assert len(types) >= 1
