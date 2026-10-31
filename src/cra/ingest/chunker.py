"""Section-aware, overlapping text chunking.

Each section is chunked independently so a chunk never spans two sections — this
keeps the ``section`` metadata accurate for citations. Within a section, text is
split into windows of roughly ``chunk_size`` characters with ``chunk_overlap``
characters of overlap, preferring to break on sentence/paragraph boundaries.
"""

from __future__ import annotations

import re

from cra.ingest.models import Chunk, Document

# Prefer breaking on paragraph, then sentence, then whitespace boundaries.
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _slugify_section(section: str) -> str:
    """Make a compact, citation-friendly section slug."""
    if not section:
        return "body"
    slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
    return slug or "body"


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows near natural boundaries."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    # Candidate boundary positions, used to snap window ends to sentence/para.
    boundaries = [m.end() for m in _BOUNDARY_RE.finditer(text)]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        target_end = min(start + chunk_size, n)
        if target_end < n:
            # Snap to the latest boundary within the window, if one exists past
            # the halfway point (avoids tiny fragments).
            candidate = max(
                (b for b in boundaries if start + chunk_size // 2 < b <= target_end),
                default=target_end,
            )
            end = candidate
        else:
            end = n
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # Advance with overlap, but always make forward progress.
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk a single document section by section."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for section, text in doc.sections:
        slug = _slugify_section(section)
        pieces = _split_text(text, chunk_size, chunk_overlap)
        for ordinal, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{slug}::{ordinal}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=slug,
                    source_url=doc.source_url,
                    text=piece,
                    ordinal=ordinal,
                )
            )
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk a list of documents."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return out
