"""Proves the typed graph-state contract: evidence/summaries pass between
agents without loss or duplication (functional requirement #5)."""

from __future__ import annotations

from cra.graph.orchestrator import Orchestrator


def test_evidence_is_deduplicated_in_state(orchestrator: Orchestrator) -> None:
    result = orchestrator.answer(
        "Compare the eligibility criteria and primary endpoints of the psoriasis trial"
    )
    state = result.state
    ids = state.evidence_chunk_ids()
    # No duplicate chunk IDs survive into the deduped evidence union.
    assert len(ids) == len(set(ids)), "evidence contains duplicate chunk IDs"


def test_summaries_reference_only_existing_evidence(orchestrator: Orchestrator) -> None:
    result = orchestrator.answer("What are the eligibility criteria for the asthma trial?")
    state = result.state
    evidence_ids = set(state.evidence_chunk_ids())
    for summary in state.summaries:
        for cid in summary.chunk_ids:
            # Every chunk a summary cites must exist in the evidence union —
            # nothing is lost or invented between research and summarize.
            assert cid in evidence_ids


def test_summaries_cover_every_sub_query(orchestrator: Orchestrator) -> None:
    result = orchestrator.answer("eligibility and endpoints for diabetes")
    state = result.state
    assert {s.sub_query for s in state.summaries} == set(state.sub_queries)


def test_state_roundtrips_through_pipeline(orchestrator: Orchestrator) -> None:
    result = orchestrator.answer("psoriasis primary endpoints")
    state = result.state
    # The terminal object is the typed contract, fully populated end-to-end.
    assert state.request_id
    assert state.question
    assert state.sub_queries
    assert state.evidence
    assert state.summaries
    assert state.answer
    # Trace has exactly one step per agent, in order.
    assert [t.agent for t in state.trace] == ["research", "summarize", "respond"]


def test_citations_are_unique_and_derived_from_evidence(
    orchestrator: Orchestrator,
) -> None:
    result = orchestrator.answer("eligibility criteria for psoriasis")
    state = result.state
    assert len(state.citations) == len(set(state.citations))
    evidence_citations = {e.citation() for e in state.evidence}
    for c in state.citations:
        assert c in evidence_citations
