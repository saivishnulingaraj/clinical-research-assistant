"""Golden-set loading and item schema."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    """One evaluation example."""

    id: str
    type: str  # answerable | unanswerable | injection
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    supporting_substrings: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    # For injection items: substrings that must NOT appear in the answer.
    forbidden_substrings: list[str] = Field(default_factory=list)


def load_golden_set(path: Path) -> list[GoldenItem]:
    """Parse a JSONL golden set into typed items."""
    items: list[GoldenItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(GoldenItem.model_validate(json.loads(line)))
    return items
