from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLMError, LLMInvalidResponseError

UNSUPPORTED_SCHEMA_KEYS = {"$schema", "$id", "default", "examples", "readOnly", "writeOnly"}


def normalize_gemini_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_gemini_schema(item)
            for key, item in value.items()
            if key not in UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [normalize_gemini_schema(item) for item in value]
    return value


def _validate(value: Any, schema: dict[str, Any], path: str = "response") -> None:
    kind = schema.get("type")
    expected = {
        "object": dict, "array": list, "string": str,
        "integer": int, "number": (int, float), "boolean": bool,
    }.get(kind)
    if expected and (not isinstance(value, expected) or kind in {"integer", "number"} and isinstance(value, bool)):
        raise LLMInvalidResponseError(f"Gemini schema validation failed: {path} must be {kind}.")
    if "enum" in schema and value not in schema["enum"]:
        raise LLMInvalidResponseError(f"Gemini schema validation failed: {path} is not an allowed value.")
    if kind == "object":
        for key in schema.get("required", []):
            if key not in value:
                raise LLMInvalidResponseError(f"Gemini schema validation failed: {path}.{key} is required.")
        for key, item in value.items():
            if key in schema.get("properties", {}):
                _validate(item, schema["properties"][key], f"{path}.{key}")
    if kind == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")


def _status_code(exc: BaseException) -> int | None:
    for candidate in (exc, getattr(exc, "response", None)):
        value = getattr(candidate, "status_code", None) or getattr(candidate, "code", None)
        if isinstance(value, int):
            return value
    return None


def _retry_delay_seconds(exc: BaseException) -> float | None:
    """Extract Google's RetryInfo without exposing the full API error."""
    message = str(exc)
    matches = (
        re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE),
        re.search(r"['\"]retryDelay['\"]:\s*['\"]([0-9]+(?:\.[0-9]+)?)s", message),
    )
    for match in matches:
        if match:
            return float(match.group(1))
    return None


@dataclass
class GeminiProvider:
    model: str
    timeout: float = 120.0
    max_retries: int = 3
    max_input_chars: int = 120_000
    client_factory: Callable[[], Any] | None = field(default=None, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    provider_name: str = field(default="Gemini", init=False)

    def _client(self) -> Any:
        if not os.getenv("GEMINI_API_KEY"):
            raise LLMError("GEMINI_API_KEY is not configured. Set it in the environment before using Gemini.")
        if self.client_factory:
            return self.client_factory()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError("Gemini support requires the 'google-genai' package.") from exc
        return genai.Client(http_options=types.HttpOptions(timeout=int(self.timeout * 1000)))

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any] | None = None,
        system_prompt: str | None = None, temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        if len(prompt) > self.max_input_chars:
            raise LLMError(
                f"Gemini request is too large ({len(prompt)} characters; limit {self.max_input_chars}). "
                "Reduce the semantic chunk size."
            )
        client = self._client()
        try:
            from google.genai import types
            config: Any = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=normalize_gemini_schema(schema) if schema else None,
                system_instruction=system_prompt,
                temperature=0 if temperature is None else temperature,
                max_output_tokens=max_output_tokens,
            )
        except ImportError:
            config = {
                "response_mime_type": "application/json",
                "response_json_schema": normalize_gemini_schema(schema) if schema else None,
                "system_instruction": system_prompt,
                "temperature": 0 if temperature is None else temperature,
                "max_output_tokens": max_output_tokens,
            }
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(model=self.model, contents=prompt, config=config)
                raw = getattr(response, "text", None)
                if not isinstance(raw, str) or not raw.strip():
                    raise LLMInvalidResponseError("Gemini returned an empty response.")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise LLMInvalidResponseError("Gemini returned malformed JSON.", raw) from exc
                if not isinstance(parsed, dict):
                    raise LLMInvalidResponseError("Gemini must return a JSON object.", raw)
                if schema:
                    try:
                        _validate(parsed, schema)
                    except LLMInvalidResponseError as exc:
                        exc.raw_response = raw
                        raise
                return parsed, raw
            except LLMInvalidResponseError:
                raise
            except Exception as exc:
                code = _status_code(exc)
                message = str(exc).lower()
                retry_delay = _retry_delay_seconds(exc)
                minute_rate_limit = code == 429 and (
                    retry_delay is not None
                    or "requestsperminute" in message
                    or "per minute" in message
                )
                if minute_rate_limit and attempt < self.max_retries:
                    self.sleep((retry_delay or min(60.0, 2**attempt)) + random.random() * 0.1)
                    continue
                if minute_rate_limit:
                    raise LLMError(
                        "Gemini per-minute rate limit remained active after retries. "
                        "Wait a minute and restart the failed stage."
                    ) from exc
                if code == 429 and ("quota" in message or "resource_exhausted" in message):
                    raise LLMError(
                        "Gemini API quota exceeded. The project may have reached its free-tier limit, "
                        "or the per-minute rate limit remained active after retries. Try again later, "
                        "select Ollama, or choose another Gemini model."
                    ) from exc
                transient = code in {429, 500, 502, 503, 504} or code is None and any(
                    word in message for word in ("timeout", "timed out", "connection", "network")
                )
                if transient and attempt < self.max_retries:
                    self.sleep(min(8.0, 0.5 * 2**attempt) + random.random() * 0.1)
                    continue
                if code in {400, 401, 403, 404} and "model" in message:
                    raise LLMError(
                        f"Gemini model '{self.model}' is unavailable. Check available models in Google AI Studio."
                    ) from exc
                raise LLMError(f"Gemini request failed: {type(exc).__name__} (HTTP {code or 'unknown'}).") from exc
        raise AssertionError("unreachable")
