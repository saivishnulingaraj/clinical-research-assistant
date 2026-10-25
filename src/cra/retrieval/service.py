"""Retrieval service: a thin, reusable layer over the FAISS vector store.

Loads the persisted index once and exposes ``retrieve`` for the research agent.
De-duplication across sub-queries lives here so the agent stays simple.
"""

from __future__ import annotations

from pathlib import Path

from cra.config import Settings
from cra.retrieval.embeddings import get_embedder
from cra.retrieval.vector_store import SearchResult, VectorStore


class RetrievalService:
    """Wraps a loaded :class:`VectorStore` with convenience helpers."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    @classmethod
    def from_settings(
        cls, settings: Settings, *, fake_embeddings: bool = False
    ) -> RetrievalService:
        embedder = get_embedder(settings.embedding_model, fake=fake_embeddings)
        store = VectorStore(embedder)
        store.load(settings.faiss_index_dir)
        return cls(store)

    @classmethod
    def from_index_dir(
        cls, index_dir: Path, settings: Settings, *, fake_embeddings: bool = False
    ) -> RetrievalService:
        embedder = get_embedder(settings.embedding_model, fake=fake_embeddings)
        store = VectorStore(embedder)
        store.load(index_dir)
        return cls(store)

    @property
    def num_chunks(self) -> int:
        return self._store.num_chunks

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """Return top-k results for a single query."""
        return self._store.search(query, top_k)

    def retrieve_many(
        self, queries: list[str], top_k: int
    ) -> tuple[list[SearchResult], dict[str, list[SearchResult]]]:
        """Retrieve for several sub-queries, returning deduped + per-query hits.

        The first element is the de-duplicated union (highest score kept for a
        given chunk_id), ordered by score descending. The second maps each query
        to its own top-k hits, which the research agent records in the trace.
        """
        per_query: dict[str, list[SearchResult]] = {}
        best: dict[str, SearchResult] = {}
        for q in queries:
            hits = self._store.search(q, top_k)
            per_query[q] = hits
            for h in hits:
                cid = h.chunk.chunk_id
                if cid not in best or h.score > best[cid].score:
                    best[cid] = h
        deduped = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return deduped, per_query
