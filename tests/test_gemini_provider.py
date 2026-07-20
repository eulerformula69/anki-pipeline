from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.config import LLMConfig
from src.gemini_provider import GeminiProvider, normalize_gemini_schema
from src.llm import LLMError, LLMInvalidResponseError
from src.llm_factory import create_llm_provider
from src.ollama_client import OllamaClient


class APIError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = code


class Models:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(text=result)


def provider(monkeypatch, results, **kwargs):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    models = Models(results)
    client = SimpleNamespace(models=models)
    value = GeminiProvider(
        "gemini-test", client_factory=lambda: client, sleep=lambda _: None, **kwargs
    )
    return value, models


def test_factory_selects_both_providers(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert isinstance(create_llm_provider(LLMConfig(provider="ollama")), OllamaClient)
    assert isinstance(create_llm_provider(LLMConfig(provider="gemini")), GeminiProvider)


def test_missing_key_fails_before_sdk_call(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    value = GeminiProvider("gemini-test", client_factory=lambda: pytest.fail("no call"))
    with pytest.raises(LLMError, match="GEMINI_API_KEY is not configured"):
        value.generate_json("hi", schema={"type": "object"})


def test_structured_response_and_sdk_parameters(monkeypatch) -> None:
    value, models = provider(monkeypatch, ['{"items": ["ok"]}'])
    schema = {
        "$schema": "ignored", "type": "object", "required": ["items"],
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    }
    parsed, raw = value.generate_json("prompt", schema=schema, max_output_tokens=50)
    assert parsed == {"items": ["ok"]} and raw.startswith("{")
    call = models.calls[0]
    assert call["model"] == "gemini-test" and call["contents"] == "prompt"
    assert "$schema" not in normalize_gemini_schema(schema)
    assert call["config"] is not None


@pytest.mark.parametrize("raw", ["", "not-json", "[]"])
def test_invalid_responses(monkeypatch, raw) -> None:
    value, _ = provider(monkeypatch, [raw])
    with pytest.raises(LLMInvalidResponseError):
        value.generate_json("prompt", schema={"type": "object"})


def test_schema_validation(monkeypatch) -> None:
    value, _ = provider(monkeypatch, ['{"items": 1}'])
    schema = {
        "type": "object", "required": ["items"],
        "properties": {"items": {"type": "array"}},
    }
    with pytest.raises(LLMInvalidResponseError, match="schema validation"):
        value.generate_json("prompt", schema=schema)


@pytest.mark.parametrize("code", [429, 503])
def test_transient_error_is_retried(monkeypatch, code) -> None:
    value, models = provider(monkeypatch, [APIError(code, "temporary"), "{}"], max_retries=1)
    assert value.generate_json("prompt", schema={"type": "object"})[0] == {}
    assert len(models.calls) == 2


def test_invalid_key_is_not_retried_or_logged(monkeypatch, caplog) -> None:
    value, models = provider(monkeypatch, [APIError(401, "invalid api key")], max_retries=3)
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMError):
        value.generate_json("prompt", schema={"type": "object"})
    assert len(models.calls) == 1
    assert "secret-value" not in caplog.text


def test_quota_message_and_input_limit(monkeypatch) -> None:
    value, _ = provider(monkeypatch, [APIError(429, "quota resource_exhausted")])
    with pytest.raises(LLMError, match="free-tier limit"):
        value.generate_json("prompt", schema={"type": "object"})
    value.max_input_chars = 2
    with pytest.raises(LLMError, match="too large"):
        value.generate_json("long", schema={"type": "object"})
