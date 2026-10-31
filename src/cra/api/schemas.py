"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Body for ``POST /ask``."""

    question: str = Field(..., min_length=1, max_length=4000)


class TraceStep(BaseModel):
    agent: str
    detail: str
    llm_calls: int
    duration_ms: float


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    cache_hits: int


class AskResponse(BaseModel):
    """Body returned by ``POST /ask``."""

    request_id: str
    answer: str
    citations: list[str]
    insufficient_evidence: bool
    retrieval_confidence: float
    trace: list[TraceStep]
    latency_ms: float
    tokens: TokenUsage
    estimated_cost_usd: float


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_chunks: int
    llm_provider: str
