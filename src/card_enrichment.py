from __future__ import annotations

from dataclasses import dataclass, replace
import re

from .models import CardCandidate, KnowledgeItem


@dataclass(frozen=True)
class CardProposal:
    knowledge_ids: list[str]
    card_type: str
    front: str
    back: str
    extra: str
    difficulty: int


def normalize_front(text: str) -> str:
    return " ".join(text.casefold().split())


def build_tags(items: list[KnowledgeItem], card_type: str) -> list[str]:
    tags: list[str] = []
    for value in [card_type, *(item.type for item in items)]:
        tag = "pitfall" if value == "mistake" else value
        if tag not in tags:
            tags.append(tag)
    return tags


def localize_card_text(
    front: str, back: str, extra: str, expressions: list,
    replacement_expressions: list,
) -> tuple[str, str, str]:
    for expression in sorted(
        replacement_expressions, key=lambda item: len(item.romanization), reverse=True
    ):
        if expression.romanization:
            pattern = re.compile(re.escape(expression.romanization), re.IGNORECASE)
            front = pattern.sub(expression.surface, front)
            back = pattern.sub(expression.surface, back)
    if expressions and not any(item.surface in front for item in expressions):
        primary = max(
            expressions,
            key=lambda item: (item.type == "example_sentence", len(item.surface)),
        )
        front = f"{primary.surface} — {front}"
        if len(front) > 120:
            front = front[:119].rstrip() + "…"
    if expressions and "Target expressions:" not in extra:
        labels = [
            f"{item.surface} [{item.reading}] ({item.romanization})"
            for item in expressions
        ]
        extra = f"{extra}\nTarget expressions: {'; '.join(labels)}".strip()
    return front, back, extra


def proposal_to_candidate(
    proposal: CardProposal,
    knowledge_by_id: dict[str, KnowledgeItem],
    expressions_by_knowledge: dict[str, list] | None = None,
    replacement_expressions: list | None = None,
) -> CardCandidate:
    items = [knowledge_by_id[item] for item in proposal.knowledge_ids]
    source_ids = sorted({
        segment_id for item in items
        for segment_id in item.source_transcript_segment_ids
    })
    expressions = []
    if expressions_by_knowledge:
        expressions = list({
            expression.id: expression
            for knowledge_id in proposal.knowledge_ids
            for expression in expressions_by_knowledge.get(knowledge_id, [])
        }.values())
    replacements = replacement_expressions if replacement_expressions is not None else expressions
    front, back, extra = localize_card_text(
        proposal.front, proposal.back, proposal.extra, expressions, replacements
    )
    return CardCandidate(
        card_id="", knowledge_ids=proposal.knowledge_ids,
        card_type=proposal.card_type, front=front, back=back,
        extra=extra, difficulty=proposal.difficulty,
        importance=max(item.importance for item in items),
        tags=build_tags(items, proposal.card_type),
        source_transcript_segment_ids=source_ids,
        expression_ids=[item.id for item in expressions],
    )


def deduplicate_candidates(candidates: list[CardCandidate]) -> list[CardCandidate]:
    unique: list[CardCandidate] = []
    positions: dict[str, int] = {}
    for candidate in candidates:
        key = normalize_front(candidate.front)
        if key not in positions:
            positions[key] = len(unique)
            unique.append(candidate)
            continue
        index = positions[key]
        existing = unique[index]
        unique[index] = replace(
            existing,
            knowledge_ids=list(dict.fromkeys(existing.knowledge_ids + candidate.knowledge_ids)),
            importance=max(existing.importance, candidate.importance),
            difficulty=max(existing.difficulty, candidate.difficulty),
            tags=list(dict.fromkeys(existing.tags + candidate.tags)),
            source_transcript_segment_ids=sorted(
                set(existing.source_transcript_segment_ids)
                | set(candidate.source_transcript_segment_ids)
            ),
            expression_ids=list(dict.fromkeys(
                (existing.expression_ids or []) + (candidate.expression_ids or [])
            )),
        )
    return unique


def number_candidates(candidates: list[CardCandidate]) -> list[CardCandidate]:
    return [
        replace(candidate, card_id=f"c_{index:04d}")
        for index, candidate in enumerate(candidates, start=1)
    ]
