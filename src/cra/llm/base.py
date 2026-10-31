"""LLM abstraction shared by the Bedrock adapter and the fake backend.

The abstraction is intentionally small: every backend turns a list of
chat-style messages into an :class:`LLMResponse` that also reports token usage.
Cost estimation, per-request call limits, and an LRU cache are layered on top in
:class:`CachedLLM` so they apply uniformly regardless of backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Message:
    """A single chat message. ``role`` is one of system/user/assistant."""

    role: str
    content: str


@dataclass
class LLMResponse:
    """Model output plus token accounting."""

    text: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageTracker:
    """Accumulates calls, tokens, and estimated cost across a request."""

    price_per_1k_input_usd: float
    price_per_1k_output_usd: float
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0

    def record(self, resp: LLMResponse, *, cached: bool = False) -> None:
        self.llm_calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        if cached:
            self.cache_hits += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        cost = (
            self.input_tokens / 1000.0 * self.price_per_1k_input_usd
            + self.output_tokens / 1000.0 * self.price_per_1k_output_usd
        )
        return round(cost, 6)


class LLMCallLimitError(RuntimeError):
    """Raised when a request exceeds its configured max LLM calls."""


@dataclass
class BaseLLM(ABC):
    """Base class for concrete LLM backends."""

    temperature: float = 0.0
    max_tokens: int = 1024

    @abstractmethod
    def _generate(self, messages: list[Message]) -> LLMResponse:
        """Backend-specific generation. Must report token counts."""
        raise NotImplementedError

    def invoke(self, messages: list[Message]) -> LLMResponse:
        return self._generate(messages)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when a backend omits usage."""
    return max(1, len(text) // 4)


@dataclass
class ChatScratchpad:
    """Helper to build message lists fluently."""

    messages: list[Message] = field(default_factory=list)

    def system(self, content: str) -> ChatScratchpad:
        self.messages.append(Message("system", content))
        return self

    def user(self, content: str) -> ChatScratchpad:
        self.messages.append(Message("user", content))
        return self

    def build(self) -> list[Message]:
        return list(self.messages)
