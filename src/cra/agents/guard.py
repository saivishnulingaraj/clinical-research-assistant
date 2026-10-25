"""Prompt-injection guard: neutralize instructions hidden in retrieved text.

Retrieved document text is untrusted. Even though the prompts wrap it in a
clearly delimited data block, we additionally sanitize common injection
patterns (e.g. "ignore previous instructions", "system:" role spoofing, fake
delimiters) so a poisoned document cannot steer the model. The guard never
*executes* the text — it only redacts markers and records that it acted.
"""

from __future__ import annotations

import re

# Phrases that attempt to override instructions.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the)?\s*(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard (all|any|the)?\s*(previous|prior|above)", re.I),
    re.compile(r"forget (everything|all|the above|previous)", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"\bnew instructions?\b", re.I),
    re.compile(r"\boverride\b.*\b(instruction|prompt|system)\b", re.I),
    # Role/delimiter spoofing.
    re.compile(r"^\s*(system|assistant|user)\s*:", re.I | re.M),
    re.compile(r"```+\s*(system|prompt)", re.I),
    re.compile(r"<\s*/?\s*(system|instructions?)\s*>", re.I),
]

_REDACTION = "[redacted-injection]"


def sanitize(text: str) -> tuple[str, bool]:
    """Return ``(clean_text, injection_detected)``.

    Each matched pattern is replaced with a visible redaction token so the
    model sees that content was removed rather than silently losing context.
    """
    detected = False
    clean = text
    for pattern in _INJECTION_PATTERNS:
        clean, n = pattern.subn(_REDACTION, clean)
        if n:
            detected = True
    return clean, detected


def sanitize_all(texts: list[str]) -> tuple[list[str], bool]:
    """Sanitize a list of evidence texts; report if any injection was found."""
    any_detected = False
    out: list[str] = []
    for t in texts:
        clean, detected = sanitize(t)
        any_detected = any_detected or detected
        out.append(clean)
    return out, any_detected
