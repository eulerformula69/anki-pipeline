from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib import error, request

from .llm import LLMError, LLMInvalidResponseError


class OllamaError(LLMError):
    """Base error with a user-readable message."""


class OllamaUnavailableError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


class OllamaHTTPError(OllamaError):
    pass


class OllamaInvalidResponseError(OllamaError, LLMInvalidResponseError):
    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    opener: Callable[..., Any] = field(default=request.urlopen, repr=False)
    format_schema: dict[str, Any] | None = field(default=None, repr=False)
    num_predict: int | None = field(default=None, repr=False)

    provider_name: str = field(default="Ollama", init=False)

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any] | None = None,
        system_prompt: str | None = None, temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        options: dict[str, Any] = {"temperature": 0 if temperature is None else temperature}
        token_limit = max_output_tokens if max_output_tokens is not None else self.num_predict
        if token_limit is not None:
            options["num_predict"] = token_limit
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "stream": False,
                "format": schema or self.format_schema or "json",
                "think": False,
                "options": options,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        try:
            with self.opener(http_request, timeout=self.timeout) as response:
                response_bytes = response.read()
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OllamaHTTPError(
                f"Ollama вернул HTTP {exc.code}: {details or exc.reason}"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaTimeoutError(
                f"Ollama не ответил за {self.timeout:g} сек."
            ) from exc
        except error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise OllamaTimeoutError(
                    f"Ollama не ответил за {self.timeout:g} сек."
                ) from exc
            raise OllamaUnavailableError(
                f"Ollama недоступен по адресу {self.base_url}. Запустите Ollama."
            ) from exc
        except OSError as exc:
            raise OllamaUnavailableError(
                f"Ollama недоступен по адресу {self.base_url}: {exc}"
            ) from exc

        try:
            raw_envelope = response_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OllamaInvalidResponseError(
                "API Ollama вернул ответ не в UTF-8.",
                response_bytes.decode("utf-8", errors="replace"),
            ) from exc

        try:
            envelope = json.loads(raw_envelope)
            model_response = envelope["response"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OllamaInvalidResponseError(
                "API Ollama вернул некорректный ответ.", raw_envelope
            ) from exc

        if not isinstance(model_response, str):
            raise OllamaInvalidResponseError(
                "В ответе Ollama отсутствует текст модели.", raw_envelope
            )
        try:
            parsed = json.loads(model_response)
        except json.JSONDecodeError as exc:
            raise OllamaInvalidResponseError(
                "Модель вернула невалидный JSON.", model_response
            ) from exc
        if not isinstance(parsed, dict):
            raise OllamaInvalidResponseError(
                "Модель должна вернуть JSON-объект.", model_response
            )
        return parsed, model_response
