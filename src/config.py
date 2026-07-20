from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"
    ollama_model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout: float = 120.0
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_timeout: float = 120.0
    gemini_max_retries: int = 3
    gemini_max_input_chars: int = 120_000

    @classmethod
    def from_env(cls, **overrides: object) -> "LLMConfig":
        values: dict[str, object] = {
            "provider": os.getenv("LLM_PROVIDER", "ollama").lower(),
            "gemini_model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            "gemini_timeout": float(os.getenv("GEMINI_TIMEOUT_SECONDS", "120")),
            "gemini_max_retries": int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            "gemini_max_input_chars": int(os.getenv("GEMINI_MAX_INPUT_CHARS", "120000")),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        config = cls(**values)
        if config.provider not in {"ollama", "gemini"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'gemini'.")
        return config
