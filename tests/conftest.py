"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from cra.config import Settings
from cra.graph.orchestrator import Orchestrator
from cra.ingest.pipeline import run_ingest
from cra.retrieval.service import RetrievalService

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory) -> Path:
    """Build a FAISS index from the sample corpus once per test session."""
    d = tmp_path_factory.mktemp("cra_index")
    settings = Settings(chunk_size=600, chunk_overlap=100, faiss_index_dir=d)
    run_ingest(SAMPLE_DIR, settings, fake_embeddings=True, index_dir=d)
    return d


@pytest.fixture
def settings(index_dir: Path) -> Settings:
    return Settings(
        llm_provider="fake",
        chunk_size=600,
        chunk_overlap=100,
        faiss_index_dir=index_dir,
        # Fake-embedder-calibrated gate: the hashing embedder yields lower,
        # noisier scores than a real model, so widen candidates and lower the
        # confidence gate (mirrors the eval runner's fake-path settings).
        confidence_threshold=0.08,
        retrieval_top_k=8,
    )


@pytest.fixture
def orchestrator(settings: Settings, index_dir: Path) -> Orchestrator:
    retrieval = RetrievalService.from_index_dir(index_dir, settings, fake_embeddings=True)
    return Orchestrator(settings, retrieval)
