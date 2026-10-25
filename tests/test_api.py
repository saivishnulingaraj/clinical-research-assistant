"""API tests using FastAPI's TestClient against a real (fake-LLM) index."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cra.config import Settings, get_settings
from cra.ingest.pipeline import run_ingest

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    # Build an index the app will load, and point settings at it via env.
    idx = tmp_path / "index"
    build_settings = Settings(chunk_size=600, chunk_overlap=100, faiss_index_dir=idx)
    run_ingest(SAMPLE_DIR, build_settings, fake_embeddings=True, index_dir=idx)

    monkeypatch.setenv("CRA_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CRA_FAISS_INDEX_DIR", str(idx))
    monkeypatch.setenv("CRA_CONFIDENCE_THRESHOLD", "0.05")
    get_settings.cache_clear()

    # Import the app after env is set so lifespan picks up the config.
    from cra.api.app import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    os.environ.pop("CRA_FAISS_INDEX_DIR", None)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["num_chunks"] > 0
    assert body["llm_provider"] == "fake"


def test_ask_returns_full_contract(client: TestClient) -> None:
    resp = client.post(
        "/ask",
        json={"question": "What are the eligibility criteria for psoriasis?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Required response fields per the spec.
    for key in (
        "request_id",
        "answer",
        "citations",
        "trace",
        "latency_ms",
        "tokens",
        "estimated_cost_usd",
    ):
        assert key in body
    assert body["request_id"]
    assert [t["agent"] for t in body["trace"]] == ["research", "summarize", "respond"]
    assert body["tokens"]["total_tokens"] > 0
    assert "not medical advice" in body["answer"].lower()


def test_ask_redacts_pii_in_question(client: TestClient) -> None:
    resp = client.post(
        "/ask",
        json={"question": "My email is bob@example.com, what psoriasis endpoints?"},
    )
    assert resp.status_code == 200
    # The echoed/processed answer path must not surface the raw email.
    assert "bob@example.com" not in resp.text


def test_ask_validation_error_on_empty(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422
