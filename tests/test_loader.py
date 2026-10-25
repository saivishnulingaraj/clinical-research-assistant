"""Tests for document loading and front-matter/section parsing."""

from __future__ import annotations

from pathlib import Path

from cra.ingest.loader import load_document, load_documents

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_loads_all_sample_docs() -> None:
    docs = load_documents(SAMPLE_DIR)
    # 15 synthetic trials; README.md is skipped.
    assert len(docs) == 15
    doc_ids = {d.doc_id for d in docs}
    assert "SYN-0001" in doc_ids
    assert "SYN-0015" in doc_ids


def test_front_matter_and_sections() -> None:
    doc = load_document(SAMPLE_DIR / "SYN-0001.md")
    assert doc.doc_id == "SYN-0001"
    assert "Psoriasis" in doc.title
    assert doc.source_url.startswith("https://synthetic.example/")
    assert doc.synthetic is True
    headings = {h for h, _ in doc.sections}
    assert "Eligibility Criteria" in headings
    assert "Primary Endpoints" in headings


def test_document_without_headings(tmp_path: Path) -> None:
    p = tmp_path / "plain.txt"
    p.write_text("Just some flat text with no headings at all.", encoding="utf-8")
    doc = load_document(p)
    assert len(doc.sections) == 1
    assert doc.sections[0][0] == ""  # empty heading
    assert "flat text" in doc.sections[0][1]
