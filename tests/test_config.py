from __future__ import annotations

import os

import pytest

from src.config import load_local_env


def test_local_env_loads_supported_values(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    (tmp_path / ".env").write_text(
        "# local settings\nLLM_PROVIDER=gemini\nGEMINI_API_KEY=secret\nUNRELATED=value\n",
        encoding="utf-8",
    )
    assert load_local_env(tmp_path)
    assert os.environ["LLM_PROVIDER"] == "gemini"
    assert os.environ["GEMINI_API_KEY"] == "secret"
    assert "UNRELATED" not in os.environ


def test_local_env_never_overrides_system_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "system-secret")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=file-secret\n", encoding="utf-8")
    load_local_env(tmp_path)
    assert os.environ["GEMINI_API_KEY"] == "system-secret"


def test_local_env_rejects_malformed_line_without_leaking_value(tmp_path) -> None:
    (tmp_path / ".env").write_text("this is secret material\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1") as error:
        load_local_env(tmp_path)
    assert "secret material" not in str(error.value)
