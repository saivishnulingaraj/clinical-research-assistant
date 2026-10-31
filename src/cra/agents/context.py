"""Shared dependencies injected into each agent node."""

from __future__ import annotations

from dataclasses import dataclass

from cra.config import Settings
from cra.llm.base import UsageTracker
from cra.llm.cache import CachedLLM
from cra.retrieval.service import RetrievalService


@dataclass
class AgentContext:
    """Everything the agent nodes need, assembled once per request."""

    settings: Settings
    llm: CachedLLM
    retrieval: RetrievalService
    usage: UsageTracker
