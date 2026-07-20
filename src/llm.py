from __future__ import annotations

import inspect
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Base error from a structured LLM provider."""


class LLMInvalidResponseError(LLMError):
    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


class StructuredLLM(Protocol):
    model: str
    provider_name: str

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]: ...


def generate_structured(
    client: StructuredLLM, prompt: str, *, schema: dict[str, Any],
    max_output_tokens: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Call the common API while preserving simple legacy test/client adapters."""
    parameters = inspect.signature(client.generate_json).parameters
    if "schema" not in parameters:
        return client.generate_json(prompt)
    return client.generate_json(
        prompt, schema=schema, max_output_tokens=max_output_tokens
    )
