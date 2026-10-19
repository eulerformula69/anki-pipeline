from __future__ import annotations

from dataclasses import dataclass
import re

from .audio_sources import infer_audio_reading, infer_audio_text


CARD_TYPES = {"recall", "application", "discrimination", "cloze"}
SOURCE_TAG = "LLM-source"


class CardInputError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedCard:
    source_id: str
    card_type: str
    front: str
    back: str


def parse_llm_response(text: str) -> list[ParsedCard]:
    cards: list[ParsedCard] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|"):
            line = line[1:]
            if line.endswith("|"):
                line = line[:-1]
        parts = [part.strip() for part in re.split(r"(?<!\\)\|", line, maxsplit=3)]
        if len(parts) == 4 and all(re.fullmatch(r":?-{2,}:?", part) for part in parts):
            continue
        parts = [part.replace(r"\|", "|") for part in parts]
        if len(parts) != 4 or not all(parts):
            raise CardInputError(
                f"Строка {line_number}: используйте ID | type | question | answer."
            )
        source_id, card_type, front, back = parts
        card_type = card_type.casefold()
        if source_id.casefold() == "id" and card_type == "type":
            continue
        if source_id in seen:
            raise CardInputError(f"Строка {line_number}: ID {source_id!r} повторяется.")
        if card_type not in CARD_TYPES:
            expected = ", ".join(sorted(CARD_TYPES))
            raise CardInputError(
                f"Строка {line_number}: неизвестный type {card_type!r}. Допустимо: {expected}."
            )
        seen.add(source_id)
        cards.append(ParsedCard(source_id, card_type, front, back))
    if not cards:
        raise CardInputError("Ответ LLM не содержит карточек.")
    return cards


def to_project_cards(parsed: list[ParsedCard], *, start: int = 1) -> list[dict[str, object]]:
    return [
        {
            "id": f"card_{position:04d}",
            "source_id": card.source_id,
            "type": card.card_type,
            "front": card.front,
            "back": card.back,
            "extra": "",
            "audio_text": infer_audio_text(card.front, card.back),
            "audio_reading": infer_audio_reading(card.front, card.back),
            "audio": None,
            "tags": [SOURCE_TAG, card.card_type],
            "included": True,
        }
        for position, card in enumerate(parsed, start)
    ]
