"""Build the LLM stack (backend + cache + limits) from settings."""

from __future__ import annotations

from cra.config import Settings
from cra.llm.base import BaseLLM
from cra.llm.cache import CachedLLM, LRUCache
from cra.llm.fake import FakeLLM


def build_backend(settings: Settings) -> BaseLLM:
    """Instantiate the raw LLM backend selected by ``CRA_LLM_PROVIDER``."""
    provider = settings.llm_provider.lower()
    if provider == "fake":
        return FakeLLM(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            max_sub_queries=settings.max_sub_queries,
        )
    if provider == "bedrock":
        # Imported lazily so AWS deps aren't required for the fake path.
        from cra.llm.bedrock import BedrockLLM

        return BedrockLLM(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            model_id=settings.bedrock_model_id,
            region=settings.aws_region,
        )
    raise ValueError(f"unknown CRA_LLM_PROVIDER: {settings.llm_provider!r}")


def build_cached_llm(settings: Settings, cache: LRUCache | None = None) -> CachedLLM:
    """Build a cache- and limit-wrapped LLM ready for the graph to use."""
    backend = build_backend(settings)
    cache = cache or LRUCache(settings.llm_cache_size)
    return CachedLLM(backend, cache, settings.max_llm_calls_per_request)
