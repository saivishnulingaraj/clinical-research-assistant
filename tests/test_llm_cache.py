"""Tests for the LRU cache, call limits, and usage/cost accounting."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cra.llm.base import BaseLLM, LLMCallLimitError, LLMResponse, Message, UsageTracker
from cra.llm.cache import CachedLLM, LRUCache


@dataclass
class CountingLLM(BaseLLM):
    """Backend that counts how many times it actually generated."""

    calls: int = 0

    def _generate(self, messages: list[Message]) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=f"resp-{self.calls}", input_tokens=10, output_tokens=5)


def _usage() -> UsageTracker:
    return UsageTracker(price_per_1k_input_usd=0.001, price_per_1k_output_usd=0.002)


def test_identical_queries_are_cached() -> None:
    backend = CountingLLM()
    llm = CachedLLM(backend, LRUCache(16), max_calls_per_request=8)
    usage = _usage()
    msgs = [Message("user", "hello")]

    r1 = llm.invoke(msgs, usage)
    r2 = llm.invoke(msgs, usage)

    assert r1.text == r2.text  # same cached response
    assert backend.calls == 1  # backend hit only once
    assert usage.cache_hits == 1
    assert usage.llm_calls == 2  # both recorded for accounting


def test_call_limit_counts_only_non_cached() -> None:
    backend = CountingLLM()
    llm = CachedLLM(backend, LRUCache(16), max_calls_per_request=2)
    usage = _usage()

    llm.invoke([Message("user", "a")], usage)
    llm.invoke([Message("user", "b")], usage)
    # Third distinct prompt should trip the limit.
    with pytest.raises(LLMCallLimitError):
        llm.invoke([Message("user", "c")], usage)

    # A cached repeat does NOT trip the limit even after reaching it.
    cached = llm.invoke([Message("user", "a")], usage)
    assert cached.text


def test_cost_accounting() -> None:
    usage = _usage()
    usage.record(LLMResponse("x", input_tokens=1000, output_tokens=500))
    # 1000/1000*0.001 + 500/1000*0.002 = 0.001 + 0.001 = 0.002
    assert usage.estimated_cost_usd == pytest.approx(0.002)
    assert usage.total_tokens == 1500


def test_lru_eviction() -> None:
    cache = LRUCache(2)
    cache.put("a", LLMResponse("a", 1, 1))
    cache.put("b", LLMResponse("b", 1, 1))
    cache.get("a")  # touch a so b is LRU
    cache.put("c", LLMResponse("c", 1, 1))  # evicts b
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
