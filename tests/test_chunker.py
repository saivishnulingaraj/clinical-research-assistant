"""Tests for the section-aware chunker."""

from __future__ import annotations

from cra.ingest.chunker import chunk_document
from cra.ingest.models import Document


def _doc(sections: list[tuple[str, str]]) -> Document:
    return Document(
        doc_id="D1",
        title="T",
        source_url="file://x",
        sections=sections,
        synthetic=True,
    )


def test_chunks_never_span_sections() -> None:
    doc = _doc(
        [
            ("Eligibility Criteria", "A. " * 300),
            ("Primary Endpoints", "B. " * 300),
        ]
    )
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    sections = {c.section for c in chunks}
    assert sections == {"eligibility-criteria", "primary-endpoints"}
    # Each chunk's text belongs entirely to one section's letter.
    for c in chunks:
        if c.section == "eligibility-criteria":
            assert "B." not in c.text
        else:
            assert "A." not in c.text


def test_chunk_ids_are_stable_and_unique() -> None:
    doc = _doc([("Summary", "word " * 500)])
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.chunk_id.startswith("D1::summary::") for c in chunks)


def test_citation_format() -> None:
    doc = _doc([("Primary Endpoints", "short text")])
    chunk = chunk_document(doc, chunk_size=800, chunk_overlap=100)[0]
    assert chunk.citation() == "[D1:primary-endpoints]"


def test_overlap_must_be_smaller_than_size() -> None:
    doc = _doc([("S", "text")])
    try:
        chunk_document(doc, chunk_size=100, chunk_overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= size")
