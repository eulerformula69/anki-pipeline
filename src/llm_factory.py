from __future__ import annotations

from .config import LLMConfig
from .gemini_provider import GeminiProvider
from .llm import StructuredLLM
from .ollama_client import OllamaClient


def create_llm_provider(config: LLMConfig) -> StructuredLLM:
    if config.provider == "gemini":
        return GeminiProvider(
            model=config.gemini_model, timeout=config.gemini_timeout,
            max_retries=config.gemini_max_retries,
            max_input_chars=config.gemini_max_input_chars,
        )
    return OllamaClient(config.ollama_model, config.ollama_url, config.ollama_timeout)
