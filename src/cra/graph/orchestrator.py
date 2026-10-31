"""Request-level orchestration: wire dependencies and run one question.

This is the single entry point used by both the API and the eval harness, so
behaviour is identical in both. It owns the per-request :class:`UsageTracker`
and assembles the :class:`AgentContext`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from cra.agents.context import AgentContext
from cra.config import Settings
from cra.graph.pipeline import run_pipeline
from cra.graph.state import GraphState
from cra.llm.base import UsageTracker
from cra.llm.cache import CachedLLM, LRUCache
from cra.llm.factory import build_cached_llm
from cra.retrieval.service import RetrievalService


@dataclass
class AnswerResult:
    """Everything needed to build an API response or score an eval."""

    state: GraphState
    usage: UsageTracker


class Orchestrator:
    """Holds process-lifetime resources (LLM backend, cache, loaded index)."""

    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        *,
        shared_llm: CachedLLM | None = None,
        shared_cache: LRUCache | None = None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        # The LRU cache is process-global so repeat queries hit it across
        # requests. The CachedLLM wrapper is cheap and recreated per request so
        # its call-limit state is per-request (the usage tracker is too).
        self._cache = shared_cache or LRUCache(settings.llm_cache_size)
        self._shared_llm = shared_llm

    @classmethod
    def from_settings(
        cls, settings: Settings, *, fake_embeddings: bool | None = None
    ) -> Orchestrator:
        fake = (
            settings.llm_provider.lower() == "fake" if fake_embeddings is None else fake_embeddings
        )
        retrieval = RetrievalService.from_settings(settings, fake_embeddings=fake)
        return cls(settings, retrieval)

    def answer(self, question: str, *, request_id: str | None = None) -> AnswerResult:
        """Run the full pipeline for a single question."""
        rid = request_id or uuid.uuid4().hex[:12]
        usage = UsageTracker(
            price_per_1k_input_usd=self._settings.price_per_1k_input_usd,
            price_per_1k_output_usd=self._settings.price_per_1k_output_usd,
        )
        llm = self._shared_llm or build_cached_llm(self._settings, self._cache)
        ctx = AgentContext(
            settings=self._settings,
            llm=llm,
            retrieval=self._retrieval,
            usage=usage,
        )
        state = GraphState(request_id=rid, question=question)
        final = run_pipeline(state, ctx)
        return AnswerResult(state=final, usage=usage)
