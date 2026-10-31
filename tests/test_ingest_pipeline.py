"""End-to-end ingestion test using the deterministic fake embedder."""

from __future__ import annotations

from pathlib import Path

from cra.config import Settings
from cra.ingest.pipeline import run_ingest
from cra.retrieval.embeddings import get_embedder
from cra.retrieval.vector_store import VectorStore

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_ingest_builds_persisted_index(tmp_path: Path) -> None:
    settings = Settings(chunk_size=600, chunk_overlap=100)
    index_dir = tmp_path / "index"

    result = run_ingest(SAMPLE_DIR, settings, fake_embeddings=True, index_dir=index_dir)
    assert result.num_documents == 15
    assert result.num_chunks > 15  # multiple sections per doc
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "chunks.json").exists()


def test_index_roundtrip_and_search(tmp_path: Path) -> None:
    settings = Settings(chunk_size=600, chunk_overlap=100)
    index_dir = tmp_path / "index"
    run_ingest(SAMPLE_DIR, settings, fake_embeddings=True, index_dir=index_dir)

    # Reload from disk into a fresh store and query it.
    store = VectorStore(get_embedder(settings.embedding_model, fake=True))
    store.load(index_dir)
    assert store.num_chunks > 0

    results = store.search("psoriasis PASI eligibility criteria", top_k=4)
    assert results
    # The hashing embedder keys on token overlap, so SYN-0001 (psoriasis)
    # should surface among the top results.
    assert any(r.chunk.doc_id == "SYN-0001" for r in results)
    # Scores are cosine similarities in [-1, 1].
    for r in results:
        assert -1.0001 <= r.score <= 1.0001
