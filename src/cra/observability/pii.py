"""Lightweight PII redaction for inputs and logs.

This is a best-effort regex scrubber for the most common identifiers — emails,
phone numbers, and long ID-like digit sequences. It is intentionally simple and
conservative; it is NOT a substitute for a dedicated PII/PHI pipeline. It runs
on the inbound question and on anything written to the structured logs so raw
identifiers never land in log storage.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phone numbers: optional +, groups of digits separated by space/dash/dot/paren.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]?\d{3,4}[\s.-]?\d{0,4}(?!\d)"
)
# ID-like: runs of 7+ digits (SSNs, MRNs, record numbers). Checked before phone
# so pure long digit strings are tagged as IDs.
_ID_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")

_EMAIL_TAG = "[REDACTED_EMAIL]"
_PHONE_TAG = "[REDACTED_PHONE]"
_ID_TAG = "[REDACTED_ID]"


def redact(text: str) -> str:
    """Return ``text`` with emails, phones, and long ID numbers redacted."""
    if not text:
        return text
    text = _EMAIL_RE.sub(_EMAIL_TAG, text)
    text = _ID_RE.sub(_ID_TAG, text)
    text = _PHONE_RE.sub(_PHONE_TAG, text)
    return text


def redact_mapping(data: dict) -> dict:
    """Redact string values in a shallow mapping (used for log records)."""
    return {k: (redact(v) if isinstance(v, str) else v) for k, v in data.items()}
