from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .anki_export import FIELD_NAMES, MODEL_NAME, _model, prepare_card, stable_guid

DEFAULT_URL = "http://127.0.0.1:8765"


class AnkiConnectError(RuntimeError):
    pass


def validate_url(value: str) -> str:
    value = value.strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise AnkiConnectError("Проверьте адрес и порт AnkiConnect.") from exc
    if (any(character.isspace() for character in value)
            or parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path or port == 0):
        raise AnkiConnectError("Укажите адрес AnkiConnect, например http://127.0.0.1:8765.")
    return value


def load_settings(root: Path) -> dict[str, str]:
    path = root / "anki-connect.json"
    if not path.exists():
        return {"url": DEFAULT_URL, "key": ""}
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("Expected settings object")
        return {"url": validate_url(settings["url"]), "key": str(settings.get("key", ""))}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AnkiConnectError("Не удалось прочитать настройки AnkiConnect. Сохраните их заново.") from exc


def save_settings(root: Path, url: str, key: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ".anki-connect.json.tmp"
    temporary.write_text(json.dumps({"url": validate_url(url), "key": key}), encoding="utf-8")
    temporary.replace(root / "anki-connect.json")


class AnkiConnect:
    def __init__(self, url: str = DEFAULT_URL, key: str = ""):
        self.url = validate_url(url)
        self.key = key

    def call(self, action: str, **params: Any) -> Any:
        payload = {"action": action, "version": 6, "params": params}
        if self.key:
            payload["key"] = self.key
        try:
            response = httpx.post(self.url, json=payload, timeout=15, trust_env=False)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            raise AnkiConnectError(
                "Нет связи с AnkiConnect. Откройте Anki, проверьте дополнение и адрес подключения."
            ) from exc
        if not isinstance(body, dict) or "result" not in body or "error" not in body:
            raise AnkiConnectError("По этому адресу нет ответа AnkiConnect. Проверьте адрес.")
        if body["error"]:
            raise AnkiConnectError(f"AnkiConnect: {body['error']}")
        return body["result"]

    def decks(self) -> list[str]:
        result = self.call("deckNames")
        if not isinstance(result, list) or any(not isinstance(name, str) for name in result):
            raise AnkiConnectError("AnkiConnect вернул некорректный список колод.")
        return sorted(result, key=str.casefold)

    def ensure_model(self) -> None:
        if MODEL_NAME in self.call("modelNames"):
            if self.call("modelFieldNames", modelName=MODEL_NAME) != list(FIELD_NAMES):
                raise AnkiConnectError("Поля шаблона anki-pipeline в Anki изменены. Импорт остановлен.")
            return
        import genanki

        model = _model(genanki)
        self.call("createModel", modelName=MODEL_NAME, inOrderFields=list(FIELD_NAMES),
                  css=model.css, cardTemplates=[{
                      "Name": template["name"], "Front": template["qfmt"], "Back": template["afmt"]
                  } for template in model.templates])

    def import_project(self, project: dict[str, Any], directory: Path,
                       deck_name: str, tags: list[str]) -> tuple[int, int]:
        if deck_name not in self.decks():
            raise AnkiConnectError("Колоды больше нет в Anki. Обновите список и выберите колоду.")
        cards = [card for card in project["cards"] if card["included"]]
        if not cards:
            raise AnkiConnectError("Включите хотя бы одну карточку.")
        if any(not card["front"].strip() or not card["back"].strip() for card in cards):
            raise AnkiConnectError("У каждой включённой карточки должны быть вопрос и ответ.")
        prepared = [prepare_card(project, card, directory, tags) for card in cards]
        self.ensure_model()
        added = updated = 0
        try:
            for card, (values, note_tags, media) in zip(cards, prepared, strict=True):
                identity = stable_guid(project["id"], card["id"])
                marker = f"anki_pipeline::{identity}"
                existing = self.call("findNotes", query=f"(guid:{identity} or tag:{marker})")
                if len(existing) > 1:
                    raise AnkiConnectError(f"Найдено несколько копий карточки {card['id']} в Anki.")
                for filename, source in media:
                    stored = self.call("storeMediaFile", filename=filename,
                                       data=base64.b64encode(source.read_bytes()).decode("ascii"))
                    if stored != filename:
                        raise AnkiConnectError("Anki изменил имя аудиофайла. Импорт остановлен.")
                fields = dict(zip(FIELD_NAMES, values, strict=True))
                if existing:
                    note_id = existing[0]
                    info = self.call("notesInfo", notes=[note_id])
                    if not info or info[0].get("modelName") != MODEL_NAME:
                        raise AnkiConnectError("Шаблон ранее отправленной карточки изменён в Anki.")
                    self.call("updateNoteFields", note={"id": note_id, "fields": fields})
                    self.call("addTags", notes=[note_id], tags=" ".join([*note_tags, marker]))
                    self.call("changeDeck", cards=info[0]["cards"], deck=deck_name)
                    updated += 1
                else:
                    note_id = self.call("addNote", note={"deckName": deck_name, "modelName": MODEL_NAME,
                              "fields": fields, "tags": [*note_tags, marker],
                              "options": {"allowDuplicate": True}})
                    if not isinstance(note_id, int) or isinstance(note_id, bool):
                        raise AnkiConnectError("Anki не подтвердил создание карточки.")
                    added += 1
        except (AnkiConnectError, OSError) as exc:
            raise AnkiConnectError(
                f"Импорт прерван. Добавлено: {added}, обновлено: {updated}. "
                f"{exc} Повторная отправка обновит уже отправленные карточки."
            ) from exc
        return added, updated
