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

NOTEBOOKLM_PROMPT = """Create a small, high-quality set of Anki cards from the Japanese grammar material in the provided sources.

The goal is not to cover every point. Select only transferable knowledge that is genuinely useful for understanding or producing Japanese. Prefer fewer strong cards over many weak cards.

CARD SELECTION RULES

- One card must test exactly one clear retrieval target.
- Do not create cards from introductions, anecdotes, repetition, biography, or meta-commentary.
- Do not create a card merely because something was mentioned in the source.
- Avoid cards about literal meanings or terminology unless that knowledge is independently useful or necessary for understanding the grammar.
- Prefer practical usage and distinctions between similar constructions over abstract definitions.
- Do not duplicate the same knowledge across several cards.
- When uncertain whether a card is useful, omit it.
- Every claim and example must be supported by the provided sources. Do not introduce unrelated grammar.

CARD TYPES

Use exactly one of these types:

1. application

The learner must complete or use a Japanese construction in a concrete, natural context.

Requirements:

- Use a complete, natural Japanese sentence.
- Include enough context to determine the intended answer.
- The blank must have one clearly expected answer.
- If other natural answers remain possible, explicitly identify the target construction in the English instruction.
- Do not use fragments such as `減る ...` when a complete sentence such as `この町の人口は減る ...` is possible.

Good example:

`application | Complete using より: 今日はいつも ... 寒[さむ]い。 | より`

Bad example:

`application | Complete: 減[へ]る ... | 一方[いっぽう]だ`

2. discrimination

The learner must distinguish between two or more similar constructions.

Requirements:

- The question must contain competing alternatives or a minimal pair.
- A question that merely asks for the definition of one particle is recall, not discrimination.
- Prefer concrete Japanese examples over theoretical descriptions.

Good example:

`discrimination | Choose から or より for each blank: 東京 ... 大阪まで行く。／今日は昨日 ... 寒[さむ]い。 | から／より`

Bad example:

`discrimination | Which particle marks a starting point? | から`

3. recall

Use only when a fact or meaning is genuinely worth recalling independently.

Requirements:

- Do not use recall merely to increase the number of cards.
- Avoid asking for literal meanings that do not improve comprehension or usage.
- Prefer application or discrimination when the learner should know how a construction behaves.

4. cloze

Use when a short, natural sentence has one unambiguous missing element.

Requirements:

- The surrounding context must make the answer clear.
- Do not use cloze when several particles or expressions could naturally fit.
- Use only one meaningful blank per card.

JAPANESE QUALITY RULES

- Use complete and natural Japanese sentences.
- Preserve the meaning taught by the source.
- Do not write misleading English descriptions such as “the side of Sakura is pretty” for `サクラの方が綺麗だ`.
- When teaching a comparison with `方`, include or clearly imply both alternatives.
- Distinguish functions such as `一方で` for contrast and `一方だ` for continuing change.
- Add furigana to kanji as `漢字[かんじ]`.
- Do not use romaji.
- Never insert spaces inside Japanese words or particles: write `から`, not `か ら`; `より`, not `よ り`.
- Do not add decorative spaces around Japanese text.

FINAL QUALITY CHECK

Before outputting each card, silently verify:

- Is this knowledge worth remembering?
- Does the card test exactly one thing?
- Does its declared type match the actual memory task?
- Is the Japanese natural and complete?
- Is the expected answer objectively clear?
- Could another natural answer fit the blank?
- Does another card already test the same knowledge?
- Would the learner understand exactly what counts as a correct answer?

Rewrite or omit any card that fails these checks.

OUTPUT FORMAT

Output only the finished cards. Do not include explanations, headings, Markdown, commentary, or a code block.

Use exactly four fields separated by ` | `:

id | type | question | answer

Allowed types:

recall
application
discrimination
cloze

Number IDs consecutively starting from 1.

Example:

1 | application | Complete using より: 今日[きょう]はいつも ... 寒[さむ]い。 | より
2 | discrimination | Choose 一方[いっぽう]で or 一方[いっぽう]だ: この町[まち]の人口[じんこう]は減[へ]る ...。／この町[まち]は静[しず]かな ...、不便[ふべん]でもある。 | 一方[いっぽう]だ／一方[いっぽう]で
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
