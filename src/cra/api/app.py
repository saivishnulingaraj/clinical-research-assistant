"""FastAPI application exposing POST /ask and GET /health.

The orchestrator (LLM backend, cache, loaded FAISS index) is built once at
startup and reused across requests. Each request gets a unique request ID that
is set on the logging context and echoed in the response and all logs.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from cra.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    TokenUsage,
    TraceStep,
)
from cra.config import get_settings
from cra.graph.orchestrator import Orchestrator
from cra.observability.logging import configure_logging, get_logger, set_request_id
from cra.observability.pii import redact

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load settings, configure logging, and warm the orchestrator."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("cra.api")
    try:
        orchestrator = Orchestrator.from_settings(settings)
        _state["orchestrator"] = orchestrator
        _state["num_chunks"] = orchestrator._retrieval.num_chunks  # noqa: SLF001
        _state["index_loaded"] = True
        logger.info("startup: index loaded with %s chunks", _state["num_chunks"])
    except FileNotFoundError:
        # Allow the app to boot without an index so /health can report the
        # degraded state; /ask will return a clear error.
        _state["orchestrator"] = None
        _state["num_chunks"] = 0
        _state["index_loaded"] = False
        logger.warning("startup: no FAISS index found; run ingestion first")
    _state["settings"] = settings
    yield
    _state.clear()


app = FastAPI(
    title="clinical-research-assistant",
    version="0.1.0",
    description="Agentic RAG over public clinical-research documents, with cited evidence.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = _state.get("settings") or get_settings()
    return HealthResponse(
        status="ok" if _state.get("index_loaded") else "degraded",
        index_loaded=bool(_state.get("index_loaded")),
        num_chunks=int(_state.get("num_chunks", 0)),
        llm_provider=settings.llm_provider,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    request_id = uuid.uuid4().hex[:12]
    set_request_id(request_id)
    logger = get_logger("cra.api")

    orchestrator: Orchestrator | None = _state.get("orchestrator")  # type: ignore[assignment]
    if orchestrator is None:
        logger.error("ask called but no index is loaded")
        raise HTTPException(
            status_code=503,
            detail="No FAISS index loaded. Run ingestion (cra-ingest) first.",
        )

    # Redact PII from the inbound question before processing/logging.
    question = redact(req.question)
    logger.info("ask received: %s", question)

    start = time.perf_counter()
    result = orchestrator.answer(question, request_id=request_id)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    state = result.state
    usage = result.usage
    logger.info(
        "ask complete: insufficient=%s citations=%s latency_ms=%s cost_usd=%s",
        state.insufficient_evidence,
        len(state.citations),
        latency_ms,
        usage.estimated_cost_usd,
    )

    return AskResponse(
        request_id=request_id,
        answer=state.answer,
        citations=state.citations,
        insufficient_evidence=state.insufficient_evidence,
        retrieval_confidence=state.retrieval_confidence,
        trace=[
            TraceStep(
                agent=t.agent,
                detail=t.detail,
                llm_calls=t.llm_calls,
                duration_ms=t.duration_ms,
            )
            for t in state.trace
        ],
        latency_ms=latency_ms,
        tokens=TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            llm_calls=usage.llm_calls,
            cache_hits=usage.cache_hits,
        ),
        estimated_cost_usd=usage.estimated_cost_usd,
    )
