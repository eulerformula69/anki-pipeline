from __future__ import annotations

import json
import re
from pathlib import Path

from .models import CardCandidate, CardCandidateCollection, TranscriptSegment
from .semantic import save_json_atomic


class NotebookLMImportError(ValueError):
    pass


_WORD = re.compile(r"[\w']{3,}", re.UNICODE)
_JAPANESE = re.compile(r"[一-龯々〆ヵヶぁ-ゟ゠-ヿ]{1,}")
_STOPWORDS = {
    "about", "after", "answer", "are", "card", "does", "each", "for", "from",
    "have", "how", "into", "make", "question", "should", "that", "the", "this",
    "what", "when", "which", "with", "would", "your",
}

LEARNING_GOALS = {"recall", "application", "discrimination", "cloze"}

NOTEBOOKLM_PROMPT = """Create Anki cards from the important Japanese grammar taught in the provided sources.

The goal is to cover the useful grammar points and distinctions of the lesson with clear, practical cards. Do not try to make a card from every sentence, but do not omit an important grammar point just to keep the deck small.

A grammar point may have more than one card when the cards train different skills. For example, one card may test using a construction and another may test distinguishing it from a similar construction.

Avoid duplicate retrieval: two cards should not ask essentially the same question in different words.

CARD TYPES

application — use or complete a target grammar construction in a concrete Japanese context.

discrimination — choose between two or more explicitly given competing constructions. The alternatives must appear in the question.

recall — recall an important meaning, function, rule, or concept directly.

cloze — restore one target element in a natural Japanese sentence when the intended answer is clear.

CARD RULES

- Cover the important grammar points and meaningfully different uses taught in the source.
- Keep each card focused and easy to grade.
- Prefer concrete Japanese examples when testing usage.
- Use enough context to make the intended grammar clear.
- Application and discrimination cards may cover the same grammar point when they train different skills.
- Keep a core or literal meaning when the source uses it to explain or connect grammatical uses.
- Avoid trivia, introductions, anecdotes, repetition, and video meta-commentary.
- Do not create multiple cards that test essentially the same retrieval.
- Do not invent unrelated grammar or claims not supported by the source.
- Use complete, natural Japanese when practical.
- Test the whole construction when it functions as a unit. For example, prefer `一方だ` as the answer rather than giving `だ` in the question and asking only for `一方`.
- For comparisons, give enough context to make the compared alternatives understandable.
- Do not use awkward literal English translations when a natural description of the intended meaning is possible.

JAPANESE FORMAT

- Add furigana as `漢字[かんじ]`.
- Do not use romaji.
- Do not insert spaces inside Japanese words or particles.
- Preserve natural Japanese punctuation and wording.

Before outputting, silently fix cards that are ambiguous, unnatural, incorrectly typed, or genuinely redundant.

OUTPUT FORMAT

Output only the finished cards.

Use exactly:

id | type | question | answer

Allowed types:
recall
application
discrimination
cloze

Number IDs consecutively starting from 1.
"""


def _load_transcript(path: Path) -> tuple[str, list[TranscriptSegment]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        source = data["source"]
        raw_segments = data["segments"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise NotebookLMImportError("Не удалось прочитать transcript.json.") from exc
    if not isinstance(source, str) or not isinstance(raw_segments, list):
        raise NotebookLMImportError("Некорректный формат transcript.json.")
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        try:
            segment = TranscriptSegment(item["id"], float(item["start"]), float(item["end"]), item["text"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NotebookLMImportError("В transcript.json есть некорректный сегмент.") from exc
        if not isinstance(segment.id, int) or not isinstance(segment.text, str):
            raise NotebookLMImportError("В transcript.json есть некорректный сегмент.")
        segments.append(segment)
    if not segments:
        raise NotebookLMImportError("Транскрипция пуста — сопоставить карточки с видео нельзя.")
    return source, segments


def _terms(text: str) -> set[str]:
    words = {word.casefold() for word in _WORD.findall(text) if word.casefold() not in _STOPWORDS}
    japanese = {word for word in _JAPANESE.findall(text) if len(word) > 1}
    return words | japanese


def _source_ids(front: str, back: str, segments: list[TranscriptSegment]) -> list[int]:
    card_terms = _terms(f"{front} {back}")
    if not card_terms:
        return [segments[0].id]
    scored: list[tuple[int, int]] = []
    for segment in segments:
        text = segment.text.casefold()
        shared = [term for term in card_terms if term.casefold() in text]
        score = sum(min(len(term), 12) for term in shared)
        if score:
            scored.append((score, segment.id))
    if not scored:
        return [segments[0].id]
    best = max(score for score, _ in scored)
    # A single tight range makes a better image search than merging distant mentions.
    return [segment_id for score, segment_id in scored if score == best][:1]


def _read_lines(text: str) -> list[tuple[str, str, str, str]]:
    cards: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4 or not all(parts):
            raise NotebookLMImportError(
                f"Строка {line_number}: используйте id | type | question | answer."
            )
        external_id, card_type, front, back = parts
        card_type = card_type.casefold()
        if line_number == 1 and external_id.casefold() == "id" and card_type == "type":
            continue
        if external_id in seen:
            raise NotebookLMImportError(f"Строка {line_number}: повторяющийся id {external_id!r}.")
        if card_type not in LEARNING_GOALS:
            raise NotebookLMImportError(
                f"Строка {line_number}: неизвестный type {card_type!r}; "
                "ожидается recall, application, discrimination или cloze."
            )
        seen.add(external_id)
        cards.append((external_id, card_type, front, back))
    if not cards:
        raise NotebookLMImportError("В файле не найдено ни одной карточки.")
    return cards


def import_notebooklm_cards(text: str, transcript_path: Path, destination: Path) -> CardCandidateCollection:
    source, segments = _load_transcript(transcript_path)
    imported = _read_lines(text)
    candidates: list[CardCandidate] = []
    for position, (external_id, card_type, front, back) in enumerate(imported, 1):
        card_id = f"n_{position:04d}"
        candidates.append(CardCandidate(
            card_id=card_id, knowledge_ids=[f"notebooklm_{external_id}"], card_type=card_type,
            front=front, back=back, extra="", difficulty=3, importance=3,
            tags=["notebooklm", card_type], source_transcript_segment_ids=_source_ids(front, back, segments),
            expression_ids=[],
        ))
    result = CardCandidateCollection(source, "notebooklm-import-v1", candidates)
    save_json_atomic(result.to_dict(), destination)
    return result
