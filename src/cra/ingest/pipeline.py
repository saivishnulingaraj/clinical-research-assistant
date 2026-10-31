"""The ingestion pipeline: load -> chunk -> embed -> build & persist index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cra.config import Settings
from cra.ingest.chunker import chunk_documents
from cra.ingest.loader import load_documents
from cra.ingest.models import Chunk, Document
from cra.retrieval.embeddings import get_embedder
from cra.retrieval.vector_store import VectorStore


@dataclass
class IngestResult:
    """Summary of an ingestion run."""

    num_documents: int
    num_chunks: int
    index_dir: Path
    documents: list[Document]
    chunks: list[Chunk]


def run_ingest(
    source_dir: Path,
    settings: Settings,
    *,
    fake_embeddings: bool = False,
    index_dir: Path | None = None,
) -> IngestResult:
    """Run the full ingestion pipeline and persist the FAISS index.

    ``fake_embeddings=True`` selects the deterministic hashing embedder so the
    pipeline can run in tests/CI without downloading a model.
    """
    docs = load_documents(source_dir)
    if not docs:
        raise ValueError(f"no documents found under {source_dir}")

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("documents produced zero chunks")

    embedder = get_embedder(settings.embedding_model, fake=fake_embeddings)
    store = VectorStore(embedder)
    store.build(chunks)

    target = index_dir or settings.faiss_index_dir
    store.save(target)

    return IngestResult(
        num_documents=len(docs),
        num_chunks=len(chunks),
        index_dir=target,
        documents=docs,
        chunks=chunks,
    )
