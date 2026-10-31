"""Tests for the retrieval service (dedup + per-query hits)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cra.config import Settings
from cra.ingest.pipeline import run_ingest
from cra.retrieval.service import RetrievalService

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="module")
def service(tmp_path_factory) -> RetrievalService:
    index_dir = tmp_path_factory.mktemp("idx")
    settings = Settings(chunk_size=600, chunk_overlap=100, faiss_index_dir=index_dir)
    run_ingest(SAMPLE_DIR, settings, fake_embeddings=True, index_dir=index_dir)
    return RetrievalService.from_index_dir(index_dir, settings, fake_embeddings=True)


def test_single_retrieve(service: RetrievalService) -> None:
    hits = service.retrieve("psoriasis PASI eligibility", top_k=4)
    assert hits
    assert any(h.chunk.doc_id == "SYN-0001" for h in hits)


def test_retrieve_many_dedup(service: RetrievalService) -> None:
    queries = [
        "psoriasis eligibility criteria",
        "psoriasis primary endpoints PASI 75",
    ]
    deduped, per_query = service.retrieve_many(queries, top_k=4)
    assert set(per_query.keys()) == set(queries)
    # Deduped union has unique chunk IDs.
    ids = [r.chunk.chunk_id for r in deduped]
    assert len(ids) == len(set(ids))
    # Sorted by score descending.
    scores = [r.score for r in deduped]
    assert scores == sorted(scores, reverse=True)
