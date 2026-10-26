from __future__ import annotations

import json
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .audio_sources import (
    AudioDownload, AudioLookupError, find_commons_audio, find_jpod101_audio,
    validate_audio,
)
from .audio_tts import generate_audio


def read_candidates(project: Path, card: dict) -> list[dict]:
    path = project / "audio-candidates" / f"{card['id']}.json"
    if not path.exists():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["text"], manifest["reading"]) != (card["audio_text"], card["audio_reading"]):
        return []
    return manifest["items"]


def find_candidates(project: Path, card_id: str, text: str, reading: str) -> None:
    sources = [
        ("JapanesePod101", lambda: find_jpod101_audio(text, reading)),
        ("Wikimedia Commons", lambda: find_commons_audio(text)),
        ("Nanami · синтез речи", lambda: generate_audio(text)),
    ]

    def lookup(source):
        name, fetch = source
        try:
            download = fetch()
            if download is None:
                return name, None, "Запись не найдена."
            validate_audio(download.data)
            return name, download, ""
        except AudioLookupError as exc:
            return name, None, str(exc)

    directory = project / "audio-candidates"
    directory.mkdir(exist_ok=True)
    items = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lookup, sources))
    for name, download, error in results:
        item = {"name": name, "error": error}
        if download is not None:
            token = secrets.token_hex(16)
            (directory / token).write_bytes(download.data)
            item.update({key: value for key, value in asdict(download).items() if key != "data"})
            item["token"] = token
        items.append(item)
    manifest = {"text": text, "reading": reading, "items": items}
    # Keep the previous choices available until the complete new search is saved.
    path = directory / f"{card_id}.json"
    old = json.loads(path.read_text(encoding="utf-8"))["items"] if path.exists() else []
    temporary = directory / f"{card_id}.{secrets.token_hex(8)}.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    for item in old:
        if re.fullmatch(r"[0-9a-f]{32}", item.get("token", "")):
            (directory / item["token"]).unlink(missing_ok=True)


def selected_candidate(project: Path, card_id: str, token: str, text: str, reading: str) -> AudioDownload:
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise AudioLookupError("Недействительный вариант аудио.")
    items = read_candidates(project, {"id": card_id, "audio_text": text, "audio_reading": reading})
    item = next((item for item in items if item.get("token") == token), None)
    if item is None:
        raise AudioLookupError("Текст или варианты изменились. Нажмите «Найти варианты аудио» ещё раз.")
    try:
        data = (project / "audio-candidates" / token).read_bytes()
    except FileNotFoundError as exc:
        raise AudioLookupError("Вариант устарел. Найдите варианты аудио ещё раз.") from exc
    validate_audio(data)
    return AudioDownload(
        data, item["suffix"], item["source_url"], item["license_name"],
        item["attribution"], item["source_name"],
    )
