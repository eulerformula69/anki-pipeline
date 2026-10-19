from __future__ import annotations

import json
from pathlib import Path


def load_receipts(directory: Path) -> dict[str, int]:
    path = directory / "anki-sent.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or type(note_id) is not int or note_id <= 0
        for key, note_id in value.items()
    ):
        raise ValueError("Повреждён журнал отправки в Anki. Отправка остановлена.")
    return value


def save_receipts(directory: Path, receipts: dict[str, int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "anki-sent.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    temporary.replace(path)
