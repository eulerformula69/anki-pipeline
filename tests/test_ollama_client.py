import io
import json
import socket
from urllib import error

import pytest

from src.ollama_client import (
    OllamaClient,
    OllamaHTTPError,
    OllamaInvalidResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


class FakeHTTPResponse:
    def __init__(self, data: dict) -> None:
        self.body = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_valid_json_response() -> None:
    opener = lambda *args, **kwargs: FakeHTTPResponse(
        {"response": '{"segments": []}', "done": True}
    )
    client = OllamaClient(model="test", opener=opener)

    parsed, raw = client.generate_json("prompt")

    assert parsed == {"segments": []}
    assert raw == '{"segments": []}'


def test_invalid_model_json() -> None:
    opener = lambda *args, **kwargs: FakeHTTPResponse({"response": "not json"})

    with pytest.raises(OllamaInvalidResponseError, match="невалидный JSON"):
        OllamaClient(model="test", opener=opener).generate_json("prompt")


def test_unavailable_ollama() -> None:
    def opener(*args, **kwargs):
        raise error.URLError("connection refused")

    with pytest.raises(OllamaUnavailableError, match="недоступен"):
        OllamaClient(model="test", opener=opener).generate_json("prompt")


def test_ollama_timeout() -> None:
    def opener(*args, **kwargs):
        raise error.URLError(socket.timeout())

    with pytest.raises(OllamaTimeoutError, match="не ответил"):
        OllamaClient(model="test", timeout=1, opener=opener).generate_json("prompt")


def test_http_error() -> None:
    def opener(*args, **kwargs):
        raise error.HTTPError(
            "http://localhost", 500, "failure", {}, io.BytesIO(b"broken")
        )

    with pytest.raises(OllamaHTTPError, match="HTTP 500"):
        OllamaClient(model="test", opener=opener).generate_json("prompt")
