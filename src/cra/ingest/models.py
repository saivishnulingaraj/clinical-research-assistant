"""Data models shared across ingestion and retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A parsed source document with its section structure."""

    doc_id: str
    title: str
    source_url: str
    # Ordered list of (section_heading, section_text). Documents without
    # headings yield a single ("", full_text) entry.
    sections: list[tuple[str, str]]
    synthetic: bool = True


class Chunk(BaseModel):
    """A retrievable chunk with provenance metadata.

    ``chunk_id`` is stable and human-readable (``<doc_id>::<section>::<n>``) so
    it can be cited and traced back to the source text.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    source_url: str
    text: str
    # Position of the chunk within its section (0-based), useful for ordering.
    ordinal: int = Field(default=0)

    def citation(self) -> str:
        """Render the canonical citation form ``[doc_id:section]``."""
        section = self.section or "body"
        return f"[{self.doc_id}:{section}]"
