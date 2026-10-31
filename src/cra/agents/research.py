"""Research agent: decompose -> retrieve -> dedup -> write evidence to state."""

from __future__ import annotations

import re
import time

from cra.agents.context import AgentContext
from cra.agents.prompts import RESEARCH_SYSTEM, decompose_prompt
from cra.graph.state import AgentTraceStep, EvidenceItem, GraphState
from cra.llm.base import ChatScratchpad

# Generic terms that appear in most questions/docs and carry little signal for
# deciding whether a question is actually grounded in the corpus.
_GROUNDING_STOPWORDS = {
    "what",
    "which",
    "whose",
    "whom",
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "does",
    "did",
    "how",
    "why",
    "when",
    "where",
    "study",
    "trial",
    "trials",
    "primary",
    "secondary",
    "endpoint",
    "endpoints",
    "eligibility",
    "criteria",
    "intervention",
    "dose",
    "dosing",
    "schedule",
    "design",
    "phase",
    "use",
    "uses",
    "used",
    "its",
    "their",
    "compare",
    "between",
    "both",
    "report",
    "reported",
    "across",
    "all",
    "fictional",
    "country",
    "city",
    "company",
}


def _parse_sub_queries(text: str, max_sub_queries: int) -> list[str]:
    """Parse one-per-line sub-queries, tolerating bullet markers."""
    subs: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. )").strip()
        if line:
            subs.append(line)
    return subs[:max_sub_queries] if subs else []


def _salient_terms(question: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return {t for t in toks if len(t) > 3 and t not in _GROUNDING_STOPWORDS}


def _lexical_grounding(question: str, evidence: list[EvidenceItem]) -> float:
    """Fraction of the question's salient terms present in the evidence text.

    This is an embedder-agnostic grounding signal: a question about topics
    absent from the corpus (e.g. "CRISPR", a made-up place) scores near zero
    even if cosine similarity is incidentally non-trivial, which lets the
    confidence gate refuse rather than answer from unrelated chunks.
    """
    terms = _salient_terms(question)
    if not terms:
        return 1.0
    # Include both the chunk text and the document title so that a question
    # keyed on an intervention name (which may live only in the title) still
    # counts as grounded when its document was retrieved.
    corpus = " ".join(f"{e.title} {e.text}".lower() for e in evidence)
    present = sum(1 for t in terms if t in corpus)
    return present / len(terms)


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

    # 3) Confidence blends semantic similarity (mean cosine) with lexical
    #    grounding (are the question's key terms actually present in the
    #    retrieved evidence?). The grounding term makes the gate robust to weak
    #    embedders and to off-topic questions whose chunks score incidentally.
    mean_cosine = sum(e.score for e in evidence) / len(evidence) if evidence else 0.0
    grounding = _lexical_grounding(state.question, evidence) if evidence else 0.0
    # Grounding gates the cosine score: a question whose salient terms are
    # absent from the retrieved evidence (and titles) is treated as ungrounded
    # and refused, regardless of incidental similarity.
    confidence = mean_cosine * grounding

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
                f"(confidence={confidence:.3f}, cosine={mean_cosine:.3f}, "
                f"grounding={grounding:.2f})"
            ),
            llm_calls=ctx.usage.llm_calls - calls_before,
            duration_ms=round(duration_ms, 2),
        )
    )
    return state
