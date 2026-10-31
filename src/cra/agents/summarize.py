"""Summarize agent: one grounded summary per sub-query, carrying chunk IDs."""

from __future__ import annotations

import time

from cra.agents.context import AgentContext
from cra.agents.guard import sanitize
from cra.agents.prompts import SUMMARIZE_SYSTEM, summarize_prompt
from cra.graph.state import AgentTraceStep, GraphState, SubQuerySummary
from cra.llm.base import ChatScratchpad


def summarize_node(state: GraphState, ctx: AgentContext) -> GraphState:
    """Produce a summary per sub-query using only the retrieved evidence."""
    start = time.perf_counter()
    calls_before = ctx.usage.llm_calls

    # Map chunk_id -> EvidenceItem for quick lookup.
    by_id = {e.chunk_id: e for e in state.evidence}
    summaries: list[SubQuerySummary] = []

    for sub_query in state.sub_queries:
        chunk_ids = state.evidence_by_subquery.get(sub_query, [])
        # Keep only chunk_ids that survived de-duplication into evidence.
        chunk_ids = [cid for cid in chunk_ids if cid in by_id]
        if not chunk_ids:
            summaries.append(
                SubQuerySummary(
                    sub_query=sub_query,
                    summary="No evidence retrieved for this sub-query.",
                    chunk_ids=[],
                )
            )
            continue

        # Sanitize retrieved text (injection guard) before it reaches the model.
        evidence_pairs = [(cid, sanitize(by_id[cid].text)[0]) for cid in chunk_ids]
        messages = (
            ChatScratchpad()
            .system(SUMMARIZE_SYSTEM)
            .user(summarize_prompt(sub_query, evidence_pairs))
            .build()
        )
        resp = ctx.llm.invoke(messages, ctx.usage)
        summaries.append(
            SubQuerySummary(
                sub_query=sub_query,
                summary=resp.text.strip(),
                chunk_ids=chunk_ids,
            )
        )

    duration_ms = (time.perf_counter() - start) * 1000
    state.summaries = summaries
    state.trace.append(
        AgentTraceStep(
            agent="summarize",
            detail=f"{len(summaries)} sub-query summaries produced",
            llm_calls=ctx.usage.llm_calls - calls_before,
            duration_ms=round(duration_ms, 2),
        )
    )
    return state
