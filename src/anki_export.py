from __future__ import annotations

import hashlib
import html
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any


MODEL_ID = 1_876_543_220
MODEL_NAME = "anki-pipeline Basic v3"
FIELD_NAMES = ("Front", "Back", "Extra", "CardType", "Source", "Audio")
FURIGANA_TAP_SCRIPT = """
<!-- furigana-tap-toggle -->
<script>
(() => {
  if (window.furiganaTapHandler) {
    document.removeEventListener("click", window.furiganaTapHandler);
  }
  window.furiganaTapHandler = (event) => {
    const isTouchClient = document.body.classList.contains("mobile")
      || document.body.classList.contains("android")
      || document.body.classList.contains("iphone")
      || document.body.classList.contains("ipad")
      || navigator.maxTouchPoints > 0;
    if (!isTouchClient) return;
    const tappedRuby = event.target.closest("ruby");
    const wasVisible = tappedRuby?.classList.contains("furigana-visible");
    document.querySelectorAll("ruby.furigana-visible").forEach((ruby) => {
      ruby.classList.remove("furigana-visible");
    });
    if (tappedRuby && !wasVisible) tappedRuby.classList.add("furigana-visible");
  };
  document.addEventListener("click", window.furiganaTapHandler);
})();
</script>"""


class AnkiExportError(RuntimeError):
    pass


def _anki_html_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return html.escape(normalized).replace("\n", "<br>")


def stable_guid(project_id: str, card_id: str) -> str:
    value = f"anki-pipeline\0{project_id}\0{card_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def stable_deck_id(project_id: str) -> int:
    value = int.from_bytes(hashlib.sha256(project_id.encode()).digest()[:4], "big")
    return (value & 0x3FFFFFFF) | 0x40000000


def _model(genanki: Any) -> Any:
    class PipelineModel(genanki.Model):
        @property
        def _req(self) -> list[list[Any]]:
            return [[0, "all", [0]]]

    return PipelineModel(
        MODEL_ID,
        MODEL_NAME,
        fields=[{"name": name} for name in FIELD_NAMES],
        templates=[{
            "name": "Basic",
            "qfmt": '<div class="front">{{furigana:Front}}</div>' + FURIGANA_TAP_SCRIPT,
            "afmt": '{{FrontSide}}<hr id="answer"><div class="back">{{furigana:Back}}</div>'
            '{{#Extra}}<div class="extra">{{furigana:Extra}}</div>{{/Extra}}'
            '{{#Audio}}<div class="audio">{{Audio}}</div>{{/Audio}}'
            + FURIGANA_TAP_SCRIPT,
        }],
        css=""".card { font-family: Arial, 'Yu Gothic', 'Meiryo', sans-serif; font-size: 22px; text-align: left; color: #222; background: #fff; line-height: 1.45; }
.front, .back { font-size: 1.15em; }
.extra { margin-top: 1em; font-size: .85em; color: #555; }
.audio { margin-top: 1em; }
mjx-container { max-width: 100%; overflow-x: auto; overflow-y: hidden; }
ruby rt { visibility: hidden; }
ruby:hover rt { visibility: visible; }
.mobile ruby:hover rt, .android ruby:hover rt, .iphone ruby:hover rt, .ipad ruby:hover rt { visibility: hidden; }
.mobile ruby.furigana-visible rt, .android ruby.furigana-visible rt, .iphone ruby.furigana-visible rt, .ipad ruby.furigana-visible rt { visibility: visible; }""",
    )


def _note(genanki: Any, guid_value: str, model: Any, fields: list[str], tags: list[str]) -> Any:
    class StableNote(genanki.Note):
        @property
        def guid(self) -> str:
            return guid_value

    return StableNote(model=model, fields=fields, tags=tags)


def packaged_card_count(path: Path) -> int:
    try:
        with zipfile.ZipFile(path) as package:
            database_name = next(
                (name for name in ("collection.anki21b", "collection.anki2") if name in package.namelist()),
                None,
            )
            if database_name is None:
                raise AnkiExportError("В .apkg отсутствует база Anki.")
            with tempfile.TemporaryDirectory(prefix="anki-check-", dir=path.parent) as temporary:
                database_path = Path(temporary) / database_name
                database_path.write_bytes(package.read(database_name))
                database = sqlite3.connect(database_path)
                try:
                    return int(database.execute("SELECT COUNT(*) FROM cards").fetchone()[0])
                finally:
                    database.close()
    except (OSError, sqlite3.Error, zipfile.BadZipFile) as exc:
        raise AnkiExportError(f"Не удалось проверить колоду: {exc}") from exc


def build_anki_deck(
    project: dict[str, Any], destination: Path, *, deck_name: str, global_tags: list[str]
) -> Path:
    cards = [card for card in project["cards"] if card["included"]]
    if not cards:
        raise AnkiExportError("Включите хотя бы одну карточку.")
    if any(not card["front"].strip() or not card["back"].strip() for card in cards):
        raise AnkiExportError("У каждой включённой карточки должны быть вопрос и ответ.")
    try:
        import genanki
    except ImportError as exc:
        raise AnkiExportError(
            "Не установлен genanki. Выполните: py -m pip install -r requirements.txt"
        ) from exc

    project_id = project["id"]
    deck = genanki.Deck(stable_deck_id(project_id), deck_name)
    model = _model(genanki)
    media_files: list[str] = []
    for card in cards:
        tags = [
            re.sub(r"\s+", "_", tag.strip())
            for tag in dict.fromkeys([*card["tags"], *global_tags])
            if tag.strip()
        ]
        audio_field = ""
        audio = card.get("audio")
        if (
            isinstance(audio, dict)
            and audio.get("path")
            and audio.get("text", card.get("audio_text", "")) == card.get("audio_text", "")
        ):
            relative = Path(audio["path"])
            source = (destination.parent / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts:
                raise AnkiExportError(f"Некорректный путь аудио у {card['id']}.")
            if not source.is_relative_to(destination.parent.resolve()) or not source.is_file():
                raise AnkiExportError(f"Не найден аудиофайл карточки {card['id']}.")
            media_files.append(str(source))
            audio_field = f"[sound:{source.name}]"
        fields = [
            _anki_html_text(card["front"]),
            _anki_html_text(card["back"]),
            _anki_html_text(card["extra"]),
            card["type"],
            project["name"],
            audio_field,
        ]
        deck.add_note(_note(genanki, stable_guid(project_id, card["id"]), model, fields, tags))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        package = genanki.Package(deck)
        package.media_files = media_files
        package.write_to_file(str(temporary))
        count = packaged_card_count(temporary)
        if count != len(cards):
            raise AnkiExportError(f"Проверка колоды: ожидалось {len(cards)}, найдено {count}.")
        temporary.replace(destination)
    except AnkiExportError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise AnkiExportError(f"Не удалось создать колоду: {exc}") from exc
    return destination
