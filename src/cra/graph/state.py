"""Typed graph-state contract shared by the three agents.

This is the single source of truth for what flows between
``research -> summarize -> respond``. It is a Pydantic model so every field is
validated and the contract is explicit: evidence and summaries pass between
agents without loss or duplication. Each agent reads the fields it needs and
writes only the fields it owns.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from cra.ingest.models import Chunk


class EvidenceItem(BaseModel):
    """One retrieved chunk promoted into graph state with its score."""

    chunk_id: str
    doc_id: str
    section: str
    title: str
    source_url: str
    text: str
    score: float

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> EvidenceItem:
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section=chunk.section,
            title=chunk.title,
            source_url=chunk.source_url,
            text=chunk.text,
            score=score,
        )

    def citation(self) -> str:
        return f"[{self.doc_id}:{self.section or 'body'}]"


class SubQuerySummary(BaseModel):
    """A summary produced for one sub-query, with the chunks it cites."""

    sub_query: str
    summary: str
    chunk_ids: list[str] = Field(default_factory=list)


class AgentTraceStep(BaseModel):
    """One entry in the per-agent execution trace (surfaced in the API)."""

    agent: str
    detail: str
    llm_calls: int = 0
    duration_ms: float = 0.0


class GraphState(BaseModel):
    """The typed state object threaded through the LangGraph pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    # ---- input -----------------------------------------------------------
    request_id: str
    question: str

    # ---- research agent output ------------------------------------------
    sub_queries: list[str] = Field(default_factory=list)
    # De-duplicated evidence union (what summarize/respond consume).
    evidence: list[EvidenceItem] = Field(default_factory=list)
    # Per-sub-query hits, retained for the trace (may contain duplicates of
    # evidence across sub-queries — this is intentional provenance).
    evidence_by_subquery: dict[str, list[str]] = Field(default_factory=dict)
    # Mean similarity of the retained evidence; drives the confidence gate.
    retrieval_confidence: float = 0.0

    # ---- summarize agent output -----------------------------------------
    summaries: list[SubQuerySummary] = Field(default_factory=list)

    # ---- respond agent output -------------------------------------------
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False

    # ---- cross-cutting ---------------------------------------------------
    trace: list[AgentTraceStep] = Field(default_factory=list)

    def evidence_chunk_ids(self) -> list[str]:
        return [e.chunk_id for e in self.evidence]
