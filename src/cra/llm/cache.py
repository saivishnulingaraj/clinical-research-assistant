"""An LRU-cached, call-limited wrapper around any :class:`BaseLLM`.

Responsibilities:

* Cache identical prompts (an in-memory LRU keyed on the message list) so repeat
  queries do not re-hit the model. The cache is process-global and shared across
  requests, which is why it is created once at app startup.
* Enforce a per-request maximum number of *non-cached* LLM calls. Each request
  gets its own :class:`UsageTracker`; cache hits do not count against the limit.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import Lock

from cra.llm.base import (
    BaseLLM,
    LLMCallLimitError,
    LLMResponse,
    Message,
    UsageTracker,
)


def _key(messages: list[Message], temperature: float, max_tokens: int) -> str:
    payload = {
        "messages": [(m.role, m.content) for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LRUCache:
    """A tiny thread-safe LRU cache for :class:`LLMResponse` values."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._store: OrderedDict[str, LLMResponse] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, key: str, value: LLMResponse) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


class CachedLLM:
    """Wraps a backend with an LRU cache and a per-request call limit."""

    def __init__(self, backend: BaseLLM, cache: LRUCache, max_calls_per_request: int):
        self._backend = backend
        self._cache = cache
        self._max_calls = max_calls_per_request

    def invoke(
        self,
        messages: list[Message],
        usage: UsageTracker,
    ) -> LLMResponse:
        """Invoke the backend with caching and limit enforcement.

        ``usage`` tracks this request's calls/tokens/cost. A cache hit still
        records token usage (so cost reflects what the answer "cost" to produce)
        but does not count toward the non-cached call budget.
        """
        key = _key(messages, self._backend.temperature, self._backend.max_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            usage.record(cached, cached=True)
            return cached

        # Count only non-cached calls against the budget.
        non_cached_calls = usage.llm_calls - usage.cache_hits
        if non_cached_calls >= self._max_calls:
            raise LLMCallLimitError(f"exceeded max LLM calls per request ({self._max_calls})")

        resp = self._backend.invoke(messages)
        self._cache.put(key, resp)
        usage.record(resp, cached=False)
        return resp
