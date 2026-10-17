from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .card_input import CardInputError, parse_llm_response, to_project_cards
from .audio_sources import AudioDownload, infer_audio_reading, infer_audio_text


PROJECT_ID = re.compile(r"^[A-Za-z0-9А-Яа-яЁё_-]{1,80}$")
CARD_ID = re.compile(r"^card_\d{4,}$")


class ProjectError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9А-Яа-яЁё_-]+", "-", value).strip("-")[:55]
    return result or "cards"


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(self, name: str, text: str) -> dict[str, Any]:
        clean_name = " ".join(name.split())
        if len(clean_name) > 120:
            raise ProjectError("Название должно содержать не больше 120 символов.")
        try:
            parsed = parse_llm_response(text)
        except CardInputError as exc:
            raise ProjectError(str(exc)) from exc
        suffix = secrets.token_hex(4)
        if not clean_name:
            clean_name = f"{datetime.now().astimezone():%Y-%m-%d}-{suffix}"
            project_id = clean_name
        else:
            project_id = f"{_slug(clean_name)}-{suffix}"
        now = _now()
        project = {
            "version": 1,
            "id": project_id,
            "name": clean_name,
            "created_at": now,
            "updated_at": now,
            "cards": to_project_cards(parsed),
        }
        self.save(project)
        return project

    def list(self) -> list[dict[str, Any]]:
        projects = []
        for path in self.root.glob("*/project.json"):
            try:
                projects.append(self._read(path))
            except ProjectError:
                continue
        return sorted(projects, key=lambda item: item["updated_at"], reverse=True)

    def load(self, project_id: str) -> dict[str, Any]:
        return self._read(self._path(project_id))

    def save(self, project: dict[str, Any]) -> None:
        self._validate(project)
        project["updated_at"] = _now()
        path = self._path(project["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd, temporary = tempfile.mkstemp(prefix=".project-", suffix=".json", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(project, stream, ensure_ascii=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def append(self, project_id: str, text: str) -> dict[str, Any]:
        project = self.load(project_id)
        try:
            parsed = parse_llm_response(text)
        except CardInputError as exc:
            raise ProjectError(str(exc)) from exc
        project["cards"].extend(to_project_cards(parsed, start=len(project["cards"]) + 1))
        self.save(project)
        return project

    def update_card(self, project_id: str, card_id: str, updates: dict[str, Any]) -> None:
        project = self.load(project_id)
        card = self._card(project, card_id)
        for field in ("front", "back", "extra"):
            value = updates.get(field)
            if not isinstance(value, str):
                raise ProjectError(f"Поле {field} должно быть текстом.")
            card[field] = value
        audio_text = updates.get("audio_text", card.get("audio_text", ""))
        if not isinstance(audio_text, str):
            raise ProjectError("Текст для аудио должен быть строкой.")
        card["audio_text"] = audio_text.strip()
        audio_reading = updates.get("audio_reading", card.get("audio_reading", ""))
        if not isinstance(audio_reading, str):
            raise ProjectError("Чтение для аудио должно быть строкой.")
        card["audio_reading"] = audio_reading.strip()
        tags = updates.get("tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ProjectError("Теги должны быть списком строк.")
        card["tags"] = list(dict.fromkeys(tag for tag in tags if tag))
        card["included"] = bool(updates.get("included"))
        self.save(project)

    def add_card(self, project_id: str) -> str:
        project = self.load(project_id)
        position = max((int(card["id"].split("_")[1]) for card in project["cards"]), default=0) + 1
        card_id = f"card_{position:04d}"
        project["cards"].append({
            "id": card_id,
            "source_id": "manual",
            "type": "recall",
            "front": "",
            "back": "",
            "extra": "",
            "audio_text": "",
            "audio_reading": "",
            "audio": None,
            "tags": ["manual"],
            "included": True,
        })
        self.save(project)
        return card_id

    def move_card(self, project_id: str, card_id: str, direction: str) -> None:
        project = self.load(project_id)
        index = next((i for i, card in enumerate(project["cards"]) if card["id"] == card_id), None)
        if index is None:
            raise ProjectError("Карточка не найдена.")
        if direction not in {"up", "down"}:
            raise ProjectError("Неизвестное направление.")
        target = index - 1 if direction == "up" else index + 1
        if 0 <= target < len(project["cards"]):
            project["cards"][index], project["cards"][target] = project["cards"][target], project["cards"][index]
            self.save(project)

    def attach_audio(
        self, project_id: str, card_id: str, download: AudioDownload, *, source: str
    ) -> None:
        if download.suffix not in {".mp3", ".ogg", ".oga", ".wav"} or not download.data:
            raise ProjectError("Некорректный аудиофайл.")
        project = self.load(project_id)
        card = self._card(project, card_id)
        media_directory = self._path(project_id).parent / "media"
        media_directory.mkdir(parents=True, exist_ok=True)
        destination = media_directory / f"{card_id}{download.suffix}"
        temporary = media_directory / f".{card_id}{download.suffix}.tmp"
        temporary.write_bytes(download.data)
        temporary.replace(destination)
        previous = card.get("audio")
        if isinstance(previous, dict) and previous.get("path") != f"media/{destination.name}":
            old_path = self._path(project_id).parent / str(previous.get("path", ""))
            if old_path.resolve().is_relative_to(media_directory.resolve()):
                old_path.unlink(missing_ok=True)
        card["audio"] = {
            "path": f"media/{destination.name}",
            "text": card["audio_text"],
            "reading": card["audio_reading"],
            "source": source,
            "source_url": download.source_url,
            "license": download.license_name,
            "attribution": download.attribution,
        }
        self.save(project)

    def audio_path(self, project_id: str, card_id: str) -> Path:
        project = self.load(project_id)
        card = self._card(project, card_id)
        audio = card.get("audio")
        if not isinstance(audio, dict) or not audio.get("path"):
            raise ProjectError("Аудиозапись не найдена.")
        media_directory = (self._path(project_id).parent / "media").resolve()
        path = (self._path(project_id).parent / audio["path"]).resolve()
        if not path.is_relative_to(media_directory) or not path.is_file():
            raise ProjectError("Аудиозапись не найдена.")
        return path

    def _path(self, project_id: str) -> Path:
        if not PROJECT_ID.fullmatch(project_id):
            raise ProjectError("Некорректный идентификатор проекта.")
        return self.root / project_id / "project.json"

    @staticmethod
    def _card(project: dict[str, Any], card_id: str) -> dict[str, Any]:
        try:
            return next(card for card in project["cards"] if card["id"] == card_id)
        except StopIteration as exc:
            raise ProjectError("Карточка не найдена.") from exc

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError("Проект не найден или повреждён.") from exc
        if isinstance(value, dict) and isinstance(value.get("cards"), list):
            for card in value["cards"]:
                if isinstance(card, dict):
                    card.setdefault(
                        "audio_text",
                        infer_audio_text(str(card.get("front", "")), str(card.get("back", ""))),
                    )
                    card.setdefault(
                        "audio_reading",
                        infer_audio_reading(
                            str(card.get("front", "")), str(card.get("back", ""))
                        ),
                    )
                    card.setdefault("audio", None)
                    if isinstance(card["audio"], dict):
                        card["audio"].setdefault("text", card["audio_text"])
                        card["audio"].setdefault("reading", card["audio_reading"])
        self._validate(value)
        return deepcopy(value)

    @staticmethod
    def _validate(project: Any) -> None:
        if not isinstance(project, dict) or project.get("version") != 1:
            raise ProjectError("Некорректный формат проекта.")
        if not isinstance(project.get("id"), str) or not PROJECT_ID.fullmatch(project["id"]):
            raise ProjectError("Некорректный идентификатор проекта.")
        if not isinstance(project.get("name"), str) or not project["name"].strip():
            raise ProjectError("У проекта нет названия.")
        cards = project.get("cards")
        if not isinstance(cards, list) or not cards:
            raise ProjectError("Проект не содержит карточек.")
        ids = []
        for card in cards:
            if not isinstance(card, dict) or not isinstance(card.get("id"), str):
                raise ProjectError("Некорректная карточка.")
            ids.append(card["id"])
            if not CARD_ID.fullmatch(card["id"]):
                raise ProjectError("Некорректный идентификатор карточки.")
            if any(not isinstance(card.get(field), str) for field in ("front", "back", "extra", "type")):
                raise ProjectError(f"{card['id']}: текстовые поля повреждены.")
            if not isinstance(card.get("audio_text"), str):
                raise ProjectError(f"{card['id']}: текст для аудио повреждён.")
            if not isinstance(card.get("audio_reading"), str):
                raise ProjectError(f"{card['id']}: чтение для аудио повреждено.")
            audio = card.get("audio")
            if audio is not None and (
                not isinstance(audio, dict)
                or any(not isinstance(audio.get(field), str) for field in (
                    "path", "text", "reading", "source", "source_url", "license", "attribution"
                ))
            ):
                raise ProjectError(f"{card['id']}: данные аудио повреждены.")
            if not isinstance(card.get("included"), bool):
                raise ProjectError(f"{card['id']}: некорректный флаг включения.")
            if not isinstance(card.get("tags"), list) or any(not isinstance(tag, str) for tag in card["tags"]):
                raise ProjectError(f"{card['id']}: некорректные теги.")
        if len(ids) != len(set(ids)):
            raise ProjectError("Идентификаторы карточек должны быть уникальными.")
