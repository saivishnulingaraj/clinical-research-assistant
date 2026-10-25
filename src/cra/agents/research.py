"""Research agent: decompose -> retrieve -> dedup -> write evidence to state."""

from __future__ import annotations

import time

from cra.agents.context import AgentContext
from cra.agents.prompts import RESEARCH_SYSTEM, decompose_prompt
from cra.graph.state import AgentTraceStep, EvidenceItem, GraphState
from cra.llm.base import ChatScratchpad


def _parse_sub_queries(text: str, max_sub_queries: int) -> list[str]:
    """Parse one-per-line sub-queries, tolerating bullet markers."""
    subs: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. )").strip()
        if line:
            subs.append(line)
    return subs[:max_sub_queries] if subs else []


def research_node(state: GraphState, ctx: AgentContext) -> GraphState:
    """Decompose the question, retrieve per sub-query, and dedup evidence."""
    start = time.perf_counter()
    calls_before = ctx.usage.llm_calls
    max_subs = ctx.settings.max_sub_queries

    # 1) Decompose into sub-queries (bounded by config).
    messages = (
        ChatScratchpad()
        .system(RESEARCH_SYSTEM)
        .user(decompose_prompt(state.question, max_subs))
        .build()
    )
    resp = ctx.llm.invoke(messages, ctx.usage)
    sub_queries = _parse_sub_queries(resp.text, max_subs)
    if not sub_queries:
        sub_queries = [state.question]

    # 2) Retrieve per sub-query with de-duplication.
    deduped, per_query = ctx.retrieval.retrieve_many(sub_queries, ctx.settings.retrieval_top_k)

    evidence = [EvidenceItem.from_chunk(r.chunk, r.score) for r in deduped]
    evidence_by_subquery = {q: [h.chunk.chunk_id for h in hits] for q, hits in per_query.items()}

    # 3) Confidence = mean score of retained evidence (cosine in [-1, 1]).
    confidence = sum(e.score for e in evidence) / len(evidence) if evidence else 0.0

    duration_ms = (time.perf_counter() - start) * 1000
    state.sub_queries = sub_queries
    state.evidence = evidence
    state.evidence_by_subquery = evidence_by_subquery
    state.retrieval_confidence = round(confidence, 4)
    state.trace.append(
        AgentTraceStep(
            agent="research",
            detail=(
                f"{len(sub_queries)} sub-queries -> {len(evidence)} unique chunks "
                f"(confidence={confidence:.3f})"
            ),
            llm_calls=ctx.usage.llm_calls - calls_before,
            duration_ms=round(duration_ms, 2),
        )
    )
    return state
