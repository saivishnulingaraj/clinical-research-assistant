"""Tests for PII redaction."""

from __future__ import annotations

from cra.observability.pii import redact


def test_redacts_email() -> None:
    out = redact("Contact jane.doe+trial@example.com for details.")
    assert "jane.doe" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redacts_phone() -> None:
    out = redact("Call me at +1 (415) 555-0132 tomorrow.")
    assert "555-0132" not in out
    assert "[REDACTED_PHONE]" in out


def test_redacts_long_id() -> None:
    out = redact("Patient MRN 123456789 enrolled.")
    assert "123456789" not in out
    assert "[REDACTED_ID]" in out


def test_preserves_clinical_numbers() -> None:
    # Short numbers like doses and percentages must survive.
    text = "PASI 75 at week 16; 48% response; dose 20 mg."
    out = redact(text)
    assert "PASI 75" in out
    assert "48%" in out
    assert "20 mg" in out


def test_empty_input() -> None:
    assert redact("") == ""
