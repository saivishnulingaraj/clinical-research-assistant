"""Respond agent: compose the final grounded answer with citations.

Applies the confidence gate: if retrieval confidence is below the configured
threshold (or there is no evidence), it returns an explicit insufficient-evidence
response describing what little was found, rather than guessing. Every answer
carries the "not medical advice" note.
"""

from __future__ import annotations

import time

from cra.agents.context import AgentContext
from cra.agents.guard import sanitize
from cra.agents.prompts import NOT_MEDICAL_ADVICE, RESPOND_SYSTEM, respond_prompt
from cra.graph.state import AgentTraceStep, GraphState
from cra.llm.base import ChatScratchpad


def _unique_citations(citations: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def respond_node(state: GraphState, ctx: AgentContext) -> GraphState:
    """Compose the answer or refuse when confidence is too low."""
    start = time.perf_counter()
    calls_before = ctx.usage.llm_calls
    threshold = ctx.settings.confidence_threshold

    citations = _unique_citations([e.citation() for e in state.evidence])

    # Confidence gate: refuse rather than guess.
    if not state.evidence or state.retrieval_confidence < threshold:
        found = "; ".join(c for c in citations[:5]) if citations else "nothing relevant"
        state.insufficient_evidence = True
        state.answer = (
            "Insufficient evidence to answer confidently "
            f"(retrieval confidence {state.retrieval_confidence:.2f} < "
            f"threshold {threshold:.2f}). "
            f"What was found: {found}.\n\n{NOT_MEDICAL_ADVICE}"
        )
        state.citations = citations
    else:
        summary_pairs = [(s.sub_query, s.summary) for s in state.summaries]
        evidence_pairs = [(e.chunk_id, sanitize(e.text)[0]) for e in state.evidence]
        messages = (
            ChatScratchpad()
            .system(RESPOND_SYSTEM)
            .user(respond_prompt(state.question, summary_pairs, evidence_pairs))
            .build()
        )
        resp = ctx.llm.invoke(messages, ctx.usage)
        answer = resp.text.strip()
        # Guarantee the not-medical-advice note is present.
        if NOT_MEDICAL_ADVICE not in answer:
            answer = f"{answer}\n\n{NOT_MEDICAL_ADVICE}"
        state.answer = answer
        state.citations = citations
        state.insufficient_evidence = False

    duration_ms = (time.perf_counter() - start) * 1000
    state.trace.append(
        AgentTraceStep(
            agent="respond",
            detail=(
                "insufficient_evidence"
                if state.insufficient_evidence
                else f"answer composed with {len(citations)} citations"
            ),
            llm_calls=ctx.usage.llm_calls - calls_before,
            duration_ms=round(duration_ms, 2),
        )
    )
    return state
