from __future__ import annotations

import json
import re
from pathlib import Path

from .models import CardCandidate, CardCandidateCollection, TranscriptSegment
from .prompt_policy import LEARNING_CONTENT_POLICY
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

NOTEBOOKLM_PROMPT = f"""You are selecting and designing high-quality Anki cards from a source.

{LEARNING_CONTENT_POLICY}

First decide whether each possible learning item deserves a card. Fewer strong cards are
better than complete coverage. Do not retell or summarize the source, and do not create a
card for every stated idea. Each card must train exactly one clear, objectively assessable
retrieval action. Prefer application and discrimination over memorizing definitions.

For every retained item, choose exactly one learning goal:
- recall: retrieve one necessary fact, rule, term, or concise explanation;
- application: apply a rule or concept to a concrete case;
- discrimination: distinguish easily confused concepts, cases, or constructions;
- cloze: supply one unambiguous missing element from sufficient context. Use sparingly.

An application card must require actual application, not merely ask for a definition.
A discrimination card should present a meaningful contrast or minimal pair. Avoid answers
that require reproducing long lists, and never reveal the answer in the question.

For Japanese learning material, prefer concrete Japanese application examples and minimal
pairs. Use terminology cards only when the term is needed for later understanding. Use
cloze for particles or grammar only when the context permits one unambiguous answer.

Output format:
Return plain text suitable for import into an Anki pipeline.
Write one flashcard per line.
Use exactly this separator format: id | type | question | answer
Start IDs at 1 and increment by 1. Type must be recall, application, discrimination, or cloze.
Do not include a header row, numbering outside the id field, extra columns, or Markdown.
Do not use the pipe character inside a question or answer.
Write questions and answers in English.
Use kana, furigana and kanji for Japanese educating materials.
put furigana in [ ] brackets.
place space in front of word with furigana. example:  急[いそ]いでいる 時[とき]に 限[かぎ]ってバスが 遅[おく]れる。 今日[きょう]は 日本語能力試験[にほんごのうりょくしけん]を 受[う]けた。
do not use furigana for hiragana and katakana - い[い]る is prohibited
Do not use romanji for japanese
Do NOT use ANY Markdown formatting (including **, *, _, #, backticks, lists, tables).
Return the final result as a .txt file ready for upload.
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
