from __future__ import annotations

from dataclasses import dataclass, replace

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


def proposal_to_candidate(
    proposal: CardProposal,
    knowledge_by_id: dict[str, KnowledgeItem],
    expressions_by_knowledge: dict[str, list] | None = None,
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
    extra = proposal.extra
    missing_labels = [
        f"{item.surface} ({item.romanization})"
        for item in expressions
        if item.surface not in " ".join((proposal.front, proposal.back, proposal.extra))
    ]
    if missing_labels:
        extra = f"{extra}\nTarget expressions: {'; '.join(missing_labels)}".strip()
    return CardCandidate(
        card_id="", knowledge_ids=proposal.knowledge_ids,
        card_type=proposal.card_type, front=proposal.front, back=proposal.back,
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
