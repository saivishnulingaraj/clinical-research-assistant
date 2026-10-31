"""Generic OpenAI-compatible chat adapter.

Works with any provider that implements the OpenAI Chat Completions API
(``POST {base_url}/chat/completions`` with a ``Bearer`` token), e.g. AshnaAI,
OpenRouter, Groq, Together, or a local vLLM/Ollama server.

Base URL, API key, and model id are injected from :class:`Settings` — nothing
is hard-coded, and the API key is never logged. Uses ``httpx`` (already a
project dependency) so no extra SDK is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from cra.llm.base import BaseLLM, LLMResponse, Message, estimate_tokens


@dataclass
class OpenAICompatibleLLM(BaseLLM):
    """Calls an OpenAI-compatible ``/chat/completions`` endpoint."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError(
                "OpenAICompatibleLLM requires base_url, api_key, and model "
                "(set CRA_OPENAI_BASE_URL, CRA_OPENAI_API_KEY, CRA_OPENAI_MODEL)"
            )
        # Normalize: the endpoint path is appended to the configured base URL,
        # mirroring how OpenAI SDK clients treat base_url.
        self._endpoint = self.base_url.rstrip("/") + "/chat/completions"

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    )
    def _generate(self, messages: list[Message]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(self._endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return self._parse(data, messages)

    @staticmethod
    def _parse(data: dict, messages: list[Message]) -> LLMResponse:
        """Parse a Chat Completions response into an :class:`LLMResponse`."""
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("OpenAI-compatible response contained no choices")
        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()

        # Prefer real usage from the provider; fall back to an estimate so cost
        # tracking still works if a provider omits the usage block.
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0)) or estimate_tokens(
            "\n".join(m.content for m in messages)
        )
        output_tokens = int(usage.get("completion_tokens", 0)) or estimate_tokens(text)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
