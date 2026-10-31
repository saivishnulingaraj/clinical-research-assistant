"""Tests for the OpenAI-compatible LLM adapter (HTTP fully mocked)."""

from __future__ import annotations

import httpx
import pytest

from cra.config import Settings
from cra.llm.base import Message
from cra.llm.factory import build_backend
from cra.llm.openai_compatible import OpenAICompatibleLLM

# A minimal, realistic Chat Completions response body.
_OK_BODY = {
    "id": "chatcmpl-test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from Ashna."}}],
    "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
}


def _patch_transport(monkeypatch, handler) -> dict:
    """Replace httpx.Client so the adapter talks to a MockTransport."""
    captured: dict = {}

    def _record(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = request.read().decode()
        return handler(request)

    transport = httpx.MockTransport(_record)
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _factory)
    return captured


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        base_url="https://api.ashna.ai/v1/api",
        api_key="test-key-123",
        model="some-model",
        max_tokens=256,
    )


def test_happy_path_parses_text_and_usage(monkeypatch) -> None:
    captured = _patch_transport(monkeypatch, lambda req: httpx.Response(200, json=_OK_BODY))
    resp = _llm().invoke([Message("user", "hi")])
    assert resp.text == "Hello from Ashna."
    assert resp.input_tokens == 42
    assert resp.output_tokens == 7
    # Endpoint path appended to the configured base URL.
    assert captured["url"] == "https://api.ashna.ai/v1/api/chat/completions"
    # Bearer auth header is sent.
    assert captured["auth"] == "Bearer test-key-123"
    # Model + messages are in the request body.
    assert '"model":"some-model"' in captured["body"].replace(" ", "")


def test_falls_back_to_estimated_tokens_when_usage_missing(monkeypatch) -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": "abc def ghij"}}]}
    _patch_transport(monkeypatch, lambda req: httpx.Response(200, json=body))
    resp = _llm().invoke([Message("user", "a question here")])
    assert resp.text == "abc def ghij"
    # No usage block -> estimates are non-zero.
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0


def test_missing_config_raises() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleLLM(base_url="", api_key="", model="")


def test_empty_choices_raises(monkeypatch) -> None:
    _patch_transport(monkeypatch, lambda req: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ValueError):
        _llm().invoke([Message("user", "hi")])


def test_http_error_is_raised(monkeypatch) -> None:
    _patch_transport(monkeypatch, lambda req: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(httpx.HTTPStatusError):
        _llm().invoke([Message("user", "hi")])


def test_factory_builds_openai_compatible_backend() -> None:
    settings = Settings(
        llm_provider="openai_compatible",
        openai_base_url="https://api.ashna.ai/v1/api",
        openai_api_key="k",
        openai_model="m",
    )
    backend = build_backend(settings)
    assert isinstance(backend, OpenAICompatibleLLM)
    assert backend.model == "m"
