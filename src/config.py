from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_ENV_NAMES = {
    "LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL",
    "GEMINI_TIMEOUT_SECONDS", "GEMINI_MAX_RETRIES", "GEMINI_MAX_INPUT_CHARS",
}


def load_local_env(root: Path | None = None) -> bool:
    """Load supported values from .env without overriding the real environment."""
    path = (root or Path.cwd()) / ".env"
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    for number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {number}: expected NAME=value.")
        name, value = (part.strip() for part in line.split("=", 1))
        if not ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid variable name in .env line {number}.")
        if name not in SUPPORTED_ENV_NAMES:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(name, value)
    return True


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
