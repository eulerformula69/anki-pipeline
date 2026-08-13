from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .image_compression import compress_card_image
from .models import CardCandidate, TargetExpression
from .semantic import save_json_atomic

MODEL_ID = 1_876_543_219
MODEL_NAME = "anki-pipeline Basic v1"
FIELD_NAMES = (
    "Front", "Back", "Extra", "TargetText", "Reading", "Romanization",
    "Image", "Source", "StartSeconds", "Tags",
)
SAFE_CARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


class AnkiExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExportedNote:
    card_id: str
    guid: str
    knowledge_ids: list[str]
    expression_ids: list[str]
    media: str | None
    start_seconds: float | None


def _load_object(path: Path, label: str, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise AnkiExportError(f"Не найден {label}: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnkiExportError(f"Не удалось прочитать {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnkiExportError(f"{label} должен содержать JSON-объект.")
    return value


def _string_list(value: Any, label: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AnkiExportError(f"{label} должен быть массивом строк.")
    if not allow_empty and not value:
        raise AnkiExportError(f"{label} не может быть пустым.")
    if len(value) != len(set(value)):
        raise AnkiExportError(f"{label} содержит дубликаты.")
    return value


def load_card_candidates(path: Path) -> tuple[str, list[CardCandidate]]:
    data = _load_object(path, "card_candidates.json")
    assert data is not None
    source = data.get("source")
    raw_cards = data.get("card_candidates")
    if not isinstance(source, str) or not source or not isinstance(raw_cards, list):
        raise AnkiExportError("Некорректная структура card_candidates.json.")
    cards: list[CardCandidate] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_cards):
        if not isinstance(raw, dict):
            raise AnkiExportError(f"Карточка {index + 1} должна быть объектом.")
        try:
            card_id = raw["card_id"]
            front, back = raw["front"], raw["back"]
            difficulty, importance = raw["difficulty"], raw["importance"]
            source_ids = raw["source_transcript_segment_ids"]
            knowledge_ids = _string_list(raw["knowledge_ids"], f"{card_id}.knowledge_ids", allow_empty=False)
            tags = _string_list(raw["tags"], f"{card_id}.tags")
            expression_ids = _string_list(raw.get("expression_ids") or [], f"{card_id}.expression_ids")
        except (KeyError, TypeError) as exc:
            raise AnkiExportError(f"Некорректная карточка {index + 1}: {exc}") from exc
        if not isinstance(card_id, str) or not SAFE_CARD_ID.fullmatch(card_id) or card_id in ids:
            raise AnkiExportError(f"Некорректный или повторяющийся card_id: {card_id!r}.")
        if not all(isinstance(value, str) for value in (front, back, raw.get("extra", ""))):
            raise AnkiExportError(f"Текстовые поля карточки {card_id} должны быть строками.")
        if not front.strip() or not back.strip():
            raise AnkiExportError(f"Карточка {card_id} должна содержать front и back.")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (difficulty, importance)):
            raise AnkiExportError(f"difficulty/importance карточки {card_id} должны быть целыми.")
        if not isinstance(source_ids, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in source_ids):
            raise AnkiExportError(f"Некорректные source IDs карточки {card_id}.")
        ids.add(card_id)
        cards.append(CardCandidate(
            card_id, knowledge_ids, str(raw.get("card_type", "basic")), front, back,
            raw.get("extra", ""), difficulty, importance, tags, source_ids,
            expression_ids,
        ))
    if not cards:
        raise AnkiExportError("card_candidates.json не содержит карточек.")
    return source, cards


def load_verified_expressions(
    path: Path, *, expected_source: str | None = None
) -> dict[str, TargetExpression]:
    data = _load_object(path, "target_expressions.json", required=False)
    if data is None:
        return {}
    if expected_source is not None and data.get("source") != expected_source:
        raise AnkiExportError("target_expressions.json относится к другому источнику.")
    raw_items = data.get("expressions")
    if not isinstance(raw_items, list):
        raise AnkiExportError("target_expressions.json должен содержать массив expressions.")
    result: dict[str, TargetExpression] = {}
    for raw in raw_items:
        try:
            item = TargetExpression(**raw)
        except (TypeError, KeyError) as exc:
            raise AnkiExportError(f"Некорректное выражение: {exc}") from exc
        if item.verification not in {"verified", "transcribed"}:
            continue
        previous = result.get(item.id)
        if previous is None or item.confidence > previous.confidence:
            result[item.id] = item
    return result


def load_card_media(
    path: Path, output_directory: Path, *, expected_source: str | None = None
) -> dict[str, tuple[Path | None, float]]:
    data = _load_object(path, "card_media.json", required=False)
    if data is None:
        return {}
    if expected_source is not None and data.get("source") != expected_source:
        raise AnkiExportError("card_media.json относится к другому источнику.")
    raw_cards = data.get("cards")
    if not isinstance(raw_cards, list):
        raise AnkiExportError("card_media.json должен содержать массив cards.")
    result: dict[str, tuple[Path | None, float]] = {}
    for raw in raw_cards:
        if not isinstance(raw, dict) or not isinstance(raw.get("card_id"), str):
            raise AnkiExportError("Некорректная запись в card_media.json.")
        card_id = raw["card_id"]
        if card_id in result:
            raise AnkiExportError(f"Повторяющийся card_id в card_media.json: {card_id}.")
        start = raw.get("start")
        if isinstance(start, bool) or not isinstance(start, (int, float)) or start < 0:
            raise AnkiExportError(f"Некорректный start карточки {card_id}.")
        image = raw.get("image")
        if image is None:
            result[card_id] = (None, float(start))
            continue
        if not isinstance(image, dict) or not isinstance(image.get("path"), str):
            raise AnkiExportError(f"Некорректное изображение карточки {card_id}.")
        relative = Path(image["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise AnkiExportError(f"Путь изображения карточки {card_id} должен быть относительным.")
        media_path = output_directory / relative
        if not media_path.is_file():
            raise AnkiExportError(f"Не найдено изображение карточки {card_id}: {media_path}")
        result[card_id] = (media_path, float(start))
    return result


def stable_guid(source: str, card_id: str) -> str:
    digest = hashlib.sha256(f"anki-pipeline\0{source}\0{card_id}".encode("utf-8")).hexdigest()
    return digest[:20]


def stable_deck_id(source: str) -> int:
    value = int.from_bytes(hashlib.sha256(f"deck\0{source}".encode()).digest()[:4], "big")
    return (value & 0x3FFFFFFF) | 0x40000000


def _model(genanki: Any) -> Any:
    return genanki.Model(
        MODEL_ID, MODEL_NAME,
        fields=[{"name": name} for name in FIELD_NAMES],
        templates=[{
            "name": "Card 1",
            "qfmt": '<div class="front">{{furigana:Front}}</div>{{#Image}}<div class="image">{{Image}}</div>{{/Image}}',
            "afmt": '{{FrontSide}}<hr id="answer"><div class="back">{{furigana:Back}}</div>{{#Extra}}<div class="extra">{{furigana:Extra}}</div>{{/Extra}}{{#TargetText}}<div class="target">{{furigana:TargetText}}<br><span>{{Reading}}</span></div>{{/TargetText}}',
        }],
        css=""".card { font-family: Arial, 'Yu Gothic', 'Meiryo', sans-serif; font-size: 22px; text-align: left; color: #222; background: #fff; line-height: 1.45; } .front, .back { font-size: 1.15em; } .extra, .target { margin-top: 1em; font-size: .85em; color: #555; } .image { margin-top: 1em; text-align: center; } img { max-width: 100%; max-height: 55vh; }""",
    )


def _stable_note(genanki: Any, guid_value: str, model: Any, fields: list[str], tags: list[str]) -> Any:
    class StableNote(genanki.Note):
        @property
        def guid(self) -> str:
            return guid_value

    return StableNote(model=model, fields=fields, tags=tags)


def stage_media_image(source: Path, destination: Path) -> None:
    try:
        with Image.open(source) as image:
            compress_card_image(image, destination)
    except (OSError, UnidentifiedImageError) as exc:
        raise AnkiExportError(f"Не удалось подготовить изображение {source}: {exc}") from exc


def build_anki_deck(
    cards_path: Path, expressions_path: Path, media_manifest_path: Path,
    destination: Path, manifest_path: Path, *, deck_name: str | None = None,
    approved_cards_path: Path | None = None,
    global_tags: list[str] | None = None,
) -> list[ExportedNote]:
    source, cards = load_card_candidates(cards_path)
    expressions = load_verified_expressions(expressions_path, expected_source=source)
    media = load_card_media(media_manifest_path, destination.parent, expected_source=source)
    review_metadata: dict[str, Any] = {}
    field_overrides: dict[str, dict[str, str]] = {}
    common_tags = _string_list(global_tags or [], "global_tags")
    if approved_cards_path is not None:
        review = _load_object(approved_cards_path, "approved_cards.json")
        assert review is not None
        if review.get("status") != "approved" or not isinstance(review.get("approved_at"), str):
            raise AnkiExportError("Review должен быть явно одобрен перед экспортом.")
        if review.get("source") != source or not isinstance(review.get("cards"), list):
            raise AnkiExportError("approved_cards.json относится к другому источнику или повреждён.")
        originals = {card.card_id: card for card in cards}
        reviewed: list[CardCandidate] = []
        reviewed_media: dict[str, tuple[Path | None, float]] = {}
        seen: set[str] = set()
        excluded = 0
        for raw in review["cards"]:
            if not isinstance(raw, dict) or raw.get("card_id") not in originals or raw["card_id"] in seen:
                raise AnkiExportError("approved_cards.json содержит неизвестный или повторяющийся card_id.")
            card_id = raw["card_id"]
            seen.add(card_id)
            if not raw.get("included"):
                excluded += 1
                continue
            original = originals[card_id]
            tags = _string_list(raw.get("tags"), f"{card_id}.tags")
            reviewed.append(replace(
                original, front=raw.get("front"), back=raw.get("back"),
                extra=raw.get("extra", ""), tags=tags,
                difficulty=raw.get("difficulty"), importance=raw.get("importance"),
            ))
            overrides = raw.get("overrides") or {}
            if not isinstance(overrides, dict) or any(
                key not in {"target_text", "reading", "romanization"} or not isinstance(value, str)
                for key, value in overrides.items()
            ):
                raise AnkiExportError(f"Некорректные overrides карточки {card_id}.")
            field_overrides[card_id] = overrides
            image = raw.get("image") or {"mode": "none"}
            if image.get("mode") == "none":
                reviewed_media[card_id] = (None, float(media.get(card_id, (None, 0.0))[1]))
            else:
                relative = Path(str(image.get("path", "")))
                resolved = (destination.parent / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(destination.parent.resolve()) or not resolved.is_file():
                    raise AnkiExportError(f"Недопустимое изображение карточки {card_id}.")
                reviewed_media[card_id] = (resolved, float(image.get("timestamp") or 0.0))
        cards = reviewed
        media = reviewed_media
        review_metadata = {
            "review_source": approved_cards_path.name,
            "review_approved_at": review["approved_at"],
            "included_cards": len(cards), "excluded_cards": excluded,
        }
        if not cards:
            raise AnkiExportError("Review не содержит включённых карточек.")
    try:
        import genanki
    except ImportError as exc:
        raise AnkiExportError("Не установлен genanki. Выполните pip install -r requirements.txt.") from exc
    known_card_ids = {card.card_id for card in cards}
    unknown_media = set(media) - known_card_ids
    if unknown_media:
        raise AnkiExportError(f"card_media.json содержит неизвестные card IDs: {sorted(unknown_media)}.")
    deck = genanki.Deck(stable_deck_id(source), deck_name or Path(source).stem)
    model = _model(genanki)
    exported: list[ExportedNote] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="anki-export-", dir=destination.parent) as temporary:
        temp = Path(temporary)
        package_media: list[str] = []
        source_prefix = hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]
        for card in cards:
            anki_tags = [
                re.sub(r"\s+", "_", tag.strip())
                for tag in dict.fromkeys([*card.tags, *common_tags])
            ]
            if any(not tag for tag in anki_tags):
                raise AnkiExportError(f"Карточка {card.card_id} содержит пустой тег.")
            linked = [expressions[item_id] for item_id in (card.expression_ids or []) if item_id in expressions]
            target_text = "\n".join(item.surface for item in linked)
            reading = "\n".join(item.reading for item in linked)
            romanization = "\n".join(item.romanization for item in linked)
            overrides = field_overrides.get(card.card_id, {})
            target_text = overrides.get("target_text", target_text)
            reading = overrides.get("reading", reading)
            romanization = overrides.get("romanization", romanization)
            image_html = ""
            media_name: str | None = None
            start_seconds: float | None = None
            if card.card_id in media:
                media_path, start_seconds = media[card.card_id]
                if media_path is not None:
                    media_name = f"{source_prefix}_{card.card_id}.jpg"
                    staged_media = temp / media_name
                    stage_media_image(media_path, staged_media)
                    package_media.append(str(staged_media))
                    image_html = f'<img src="{html.escape(media_name, quote=True)}">'
            guid = stable_guid(source, card.card_id)

            fields = [
                html.escape(card.front), html.escape(card.back), html.escape(card.extra),
                html.escape(target_text).replace("\n", "<br>"),
                html.escape(reading).replace("\n", "<br>"),
                html.escape(romanization).replace("\n", "<br>"), image_html,
                html.escape(source), "" if start_seconds is None else f"{start_seconds:.3f}",
                html.escape(" ".join(anki_tags)),
            ]
            deck.add_note(_stable_note(genanki, guid, model, fields, anki_tags))
            exported.append(ExportedNote(
                card.card_id, guid, card.knowledge_ids,
                [item.id for item in linked], media_name, start_seconds,
            ))
        temporary_package = temp / destination.name
        package = genanki.Package(deck)
        package.media_files = package_media
        try:
            package.write_to_file(str(temporary_package))
        except Exception as exc:
            raise AnkiExportError(f"Не удалось собрать .apkg: {exc}") from exc
        if not temporary_package.is_file() or temporary_package.stat().st_size == 0:
            raise AnkiExportError("genanki не создал корректный файл колоды.")
        os.replace(temporary_package, destination)
    save_json_atomic({
        "source": source, "generator": "genanki-local-v1",
        "deck_name": deck_name or Path(source).stem,
        "deck_id": stable_deck_id(source), "model_id": MODEL_ID,
        "package": destination.name,
        "global_tags": common_tags,
        "notes": [asdict(item) for item in exported], **review_metadata,
    }, manifest_path)
    return exported
