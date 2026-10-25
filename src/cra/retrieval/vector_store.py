"""FAISS-backed vector store with on-disk persistence and chunk metadata.

The index uses inner-product over L2-normalized vectors, which is equivalent to
cosine similarity, so scores fall in ``[-1, 1]`` (typically ``[0, 1]`` for
related text). Chunk metadata is persisted alongside the index as JSON so the
store round-trips losslessly.
"""

from __future__ import annotations

import json
from pathlib import Path

from cra.ingest.models import Chunk
from cra.retrieval.embeddings import Embedder

_INDEX_FILE = "index.faiss"
_META_FILE = "chunks.json"
_INFO_FILE = "info.json"


class SearchResult:
    """A retrieved chunk paired with its similarity score."""

    __slots__ = ("chunk", "score")

    def __init__(self, chunk: Chunk, score: float) -> None:
        self.chunk = chunk
        self.score = score

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SearchResult(chunk_id={self.chunk.chunk_id!r}, score={self.score:.3f})"


class VectorStore:
    """In-memory FAISS index plus parallel chunk metadata."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index = None  # faiss.Index, created on build/load
        self._chunks: list[Chunk] = []

    # ---- construction ----------------------------------------------------
    def build(self, chunks: list[Chunk]) -> None:
        """Embed ``chunks`` and build a fresh index."""
        import faiss

        if not chunks:
            raise ValueError("cannot build an index from zero chunks")
        vectors = self._embedder.embed([c.text for c in chunks])
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index
        self._chunks = list(chunks)

    # ---- query -----------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks to ``query``."""
        if self._index is None:
            raise RuntimeError("index not built or loaded")
        if not self._chunks:
            return []
        q = self._embedder.embed([query])
        k = min(top_k, len(self._chunks))
        scores, idxs = self._index.search(q, k)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0:
                continue
            results.append(SearchResult(self._chunks[int(idx)], float(score)))
        return results

    # ---- persistence -----------------------------------------------------
    def save(self, directory: Path) -> None:
        """Persist the index and metadata under ``directory``."""
        import faiss

        if self._index is None:
            raise RuntimeError("nothing to save: index not built")
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / _INDEX_FILE))
        (directory / _META_FILE).write_text(
            json.dumps([c.model_dump() for c in self._chunks], indent=2),
            encoding="utf-8",
        )
        (directory / _INFO_FILE).write_text(
            json.dumps({"num_chunks": len(self._chunks), "dim": self._embedder.dim}),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        """Load an index and metadata previously written by :meth:`save`."""
        import faiss

        index_path = directory / _INDEX_FILE
        meta_path = directory / _META_FILE
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"no FAISS index found in {directory}; run ingestion first")
        self._index = faiss.read_index(str(index_path))
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**c) for c in raw]

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)
