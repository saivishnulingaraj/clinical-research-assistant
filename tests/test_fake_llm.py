"""Tests for the deterministic FakeLLM task protocol."""

from __future__ import annotations

from cra.llm.base import Message
from cra.llm.fake import FakeLLM


def _invoke(llm: FakeLLM, content: str) -> str:
    return llm.invoke([Message("user", content)]).text


def test_decompose_splits_compound_question() -> None:
    llm = FakeLLM(max_sub_queries=3)
    out = _invoke(
        llm,
        "TASK: decompose\nQUESTION: Compare the eligibility criteria and the "
        "primary endpoints of trial SYN-0001",
    )
    lines = [line for line in out.splitlines() if line.strip()]
    assert 1 <= len(lines) <= 3
    assert any("eligibility" in line.lower() for line in lines)


def test_summarize_carries_chunk_ids() -> None:
    llm = FakeLLM()
    out = _invoke(
        llm,
        "TASK: summarize\n[SYN-0001::eligibility-criteria::0] Adults aged 18 to 65 "
        "with plaque psoriasis. More detail follows.",
    )
    assert "SYN-0001::eligibility-criteria::0" in out


def test_respond_refuses_without_evidence() -> None:
    llm = FakeLLM()
    out = _invoke(llm, "TASK: respond\nQUESTION: What is the capital of Mars?")
    assert "insufficient evidence" in out.lower()


def test_respond_cites_doc_section() -> None:
    llm = FakeLLM()
    out = _invoke(
        llm,
        "TASK: respond\nQUESTION: eligibility criteria for psoriasis\n\n"
        "[SYN-0001::eligibility-criteria::0] Adults aged 18 to 65 years with "
        "plaque psoriasis for at least 12 months.",
    )
    assert "[SYN-0001:eligibility-criteria]" in out


def test_tokens_are_reported() -> None:
    llm = FakeLLM()
    resp = llm.invoke([Message("user", "TASK: respond\nQUESTION: hi")])
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
