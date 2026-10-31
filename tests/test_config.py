"""Tests for the settings module."""

from __future__ import annotations

import importlib

from cra.config import Settings, get_settings


def test_defaults_do_not_hardcode_bedrock_in_logic() -> None:
    """Model ID and region come from Settings, with safe defaults."""
    s = Settings()
    assert s.llm_provider == "fake"  # never calls AWS by default
    assert s.bedrock_model_id  # has a value, but is overridable via env
    assert s.aws_region


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CRA_BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
    monkeypatch.setenv("CRA_AWS_REGION", "eu-west-1")
    monkeypatch.setenv("CRA_RETRIEVAL_TOP_K", "9")
    s = Settings()
    assert s.bedrock_model_id == "amazon.nova-lite-v1:0"
    assert s.aws_region == "eu-west-1"
    assert s.retrieval_top_k == 9


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_package_imports() -> None:
    mod = importlib.import_module("cra")
    assert mod.__version__
