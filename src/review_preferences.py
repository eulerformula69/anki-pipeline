from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewPreferenceExample:
    accepted: bool
    front: str
    back: str


@dataclass(frozen=True)
class ReviewPreferenceHistory:
    examples: list[ReviewPreferenceExample]
    approved_count: int
    rejected_count: int
    project_count: int

    @property
    def is_usable(self) -> bool:
        return self.approved_count >= 3 and self.rejected_count >= 3


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _normalized_front(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold(), flags=re.UNICODE).strip()


def _valid_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned if cleaned else None


def _project_examples(directory: Path) -> list[ReviewPreferenceExample] | None:
    review = _read_object(directory / "approved_cards.json")
    candidates = _read_object(directory / "card_candidates.json")
    if (
        review is None
        or candidates is None
        or review.get("status") != "approved"
        or not isinstance(review.get("cards"), list)
        or not isinstance(candidates.get("card_candidates"), list)
    ):
        return None

    raw_by_id = {
        item.get("card_id"): item
        for item in candidates["card_candidates"]
        if isinstance(item, dict) and isinstance(item.get("card_id"), str)
    }
    examples: list[ReviewPreferenceExample] = []
    for card in review["cards"]:
        if not isinstance(card, dict) or not isinstance(card.get("included"), bool):
            continue
        raw = raw_by_id.get(card.get("card_id"), {})
        front = _valid_text(card.get("front")) or _valid_text(raw.get("front"))
        back = _valid_text(card.get("back")) or _valid_text(raw.get("back"))
        if front and back:
            examples.append(ReviewPreferenceExample(card["included"], front, back))
    return examples


def _select_diverse(
    examples: list[ReviewPreferenceExample],
    *,
    limit: int,
) -> list[ReviewPreferenceExample]:
    selected: list[ReviewPreferenceExample] = []
    seen: set[str] = set()
    for example in reversed(examples):
        normalized = _normalized_front(example.front)
        if not normalized or normalized in seen:
            continue
        selected.append(example)
        seen.add(normalized)
        if len(selected) == limit:
            break
    return list(reversed(selected))


def load_review_preference_history(
    output_root: Path,
    *,
    exclude_directory: Path | None = None,
    accepted_limit: int = 16,
    rejected_limit: int = 24,
) -> ReviewPreferenceHistory:
    accepted: list[ReviewPreferenceExample] = []
    rejected: list[ReviewPreferenceExample] = []
    project_count = 0
    excluded = exclude_directory.resolve() if exclude_directory else None
    try:
        directories = sorted(
            (item for item in output_root.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime_ns,
        )
    except OSError:
        directories = []

    for directory in directories:
        try:
            if excluded is not None and directory.resolve() == excluded:
                continue
        except OSError:
            continue
        examples = _project_examples(directory)
        if examples is None:
            continue
        project_count += 1
        accepted.extend(item for item in examples if item.accepted)
        rejected.extend(item for item in examples if not item.accepted)

    selected = [
        *_select_diverse(accepted, limit=accepted_limit),
        *_select_diverse(rejected, limit=rejected_limit),
    ]
    return ReviewPreferenceHistory(
        examples=selected,
        approved_count=len(accepted),
        rejected_count=len(rejected),
        project_count=project_count,
    )


def build_review_preference_prompt(history: ReviewPreferenceHistory) -> str:
    if not history.is_usable:
        return ""
    examples = [
        {
            "decision": "keep" if item.accepted else "discard",
            "front": item.front,
            "back": item.back,
        }
        for item in history.examples
    ]
    return f"""

PERSONAL REVIEW PREFERENCES:
The user previously kept {history.approved_count} cards and discarded
{history.rejected_count} cards across {history.project_count} completed projects.
Use the examples below to infer the user's desired learning scope.
They are preference evidence, not factual sources: never copy facts from them.
A discarded example may be redundant, malformed, too contextual, or low-value.
Prefer reusable, atomic target-language knowledge over video metadata, trivia,
isolated scene details, generic terminology, and damaged transcriptions.
When an input item resembles kept examples in learning purpose and quality, keep it.
When it resembles discarded examples, discard it. When uncertain, keep it so the
user can make the final decision.

HISTORICAL DECISIONS:
{json.dumps(examples, ensure_ascii=False, indent=2)}
"""
