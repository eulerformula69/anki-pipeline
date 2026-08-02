from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import re

from .models import CardCandidate, KnowledgeItem, KnowledgeReview


@dataclass(frozen=True)
class CardProposal:
    knowledge_ids: list[str]
    card_type: str
    front: str
    back: str
    extra: str
    difficulty: int


def fallback_card_proposals(
    items: list[KnowledgeItem], reviews: list[KnowledgeReview]
) -> list[CardProposal]:
    review_by_id = {review.knowledge_id: review for review in reviews}
    proposals: list[CardProposal] = []
    for item in items:
        if review_by_id[item.id].status == "discard":
            continue
        subject = " ".join(item.statement.split())
        front = f'What should you remember about “{subject[:76].rstrip()}”?'
        back = " ".join(
            part for part in (item.statement.strip(), item.explanation.strip()) if part
        )[:500]
        proposals.append(CardProposal(
            knowledge_ids=[item.id],
            card_type="basic",
            front=front[:120],
            back=back,
            extra="Automatic fallback: manually verify this card.",
            difficulty=3,
        ))
    return proposals


def normalize_front(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


QUESTION_STOPWORDS = {
    "a", "an", "can", "does", "for", "how", "in", "is", "of", "the",
    "to", "what", "when", "why", "with",
}


def candidates_are_duplicates(first: CardCandidate, second: CardCandidate) -> bool:
    first_front, second_front = normalize_front(first.front), normalize_front(second.front)
    if first_front == second_front:
        return True
    first_tokens = set(first_front.split()) - QUESTION_STOPWORDS
    second_tokens = set(second_front.split()) - QUESTION_STOPWORDS
    shared = first_tokens & second_tokens
    union = first_tokens | second_tokens
    front_similarity = SequenceMatcher(None, first_front, second_front).ratio()
    token_similarity = len(shared) / len(union) if union else 0.0
    first_answer = normalize_front(first.back)
    second_answer = normalize_front(second.back)
    answer_similarity = SequenceMatcher(None, first_answer, second_answer).ratio()
    return (
        (front_similarity >= 0.78 and answer_similarity >= 0.62)
        or (token_similarity >= 0.6 and answer_similarity >= 0.58)
    )


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
    extra = re.sub(
        r"(?:^|\n)Target expressions:.*?(?=\n|$)",
        "",
        extra,
    ).strip()
    for expression in replacement_expressions:
        prefix = f"{expression.surface} — "
        if front.startswith(prefix):
            front = front[len(prefix):]
    replacements = [
        (expression, spelling)
        for expression in replacement_expressions
        if (
            expression.type != "example_sentence"
            or expression.verification == "verified"
        )
        for spelling in [
            expression.romanization,
            *(expression.romanization_aliases or []),
        ]
        if spelling
    ]
    for expression, spelling in sorted(
        replacements,
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        romanization = spelling.strip(" ~〜～-")
        escaped = re.escape(romanization)
        if romanization.casefold().endswith("ii"):
            escaped = re.escape(romanization[:-1]) + "i?"
        pattern = re.compile(
            rf"(?<![A-Za-z])(?:[-~〜～]\s*)?{escaped}(?![A-Za-z])",
            re.IGNORECASE,
        )
        front = pattern.sub(expression.surface, front)
        back = pattern.sub(expression.surface, back)
        extra = pattern.sub(expression.surface, extra)
    if expressions:
        labels = [
            f"{item.surface} [{item.reading}] ({item.romanization})"
            for item in expressions
        ]
        extra = f"{extra}\nTarget expressions: {'; '.join(labels)}".strip()
    return front, back, extra


def expression_is_relevant_to_text(expression: object, text: str) -> bool:
    if getattr(expression, "type", "") != "example_sentence":
        return True
    romanization = " ".join(
        re.findall(r"[a-z]+", getattr(expression, "romanization", "").casefold())
    )
    normalized_text = " ".join(re.findall(r"[a-z]+", text.casefold()))
    return bool(romanization) and romanization in normalized_text


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
    card_text = " ".join((proposal.front, proposal.back))
    expressions = [
        item for item in expressions if expression_is_relevant_to_text(item, card_text)
    ]
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


def candidates_share_unsplit_knowledge(
    first: CardCandidate,
    second: CardCandidate,
    *,
    split_knowledge_ids: set[str],
) -> bool:
    shared_ids = set(first.knowledge_ids) & set(second.knowledge_ids)
    return bool(shared_ids - split_knowledge_ids)


def deduplicate_candidates(
    candidates: list[CardCandidate],
    *,
    split_knowledge_ids: set[str] | None = None,
) -> list[CardCandidate]:
    split_ids = split_knowledge_ids or set()
    unique: list[CardCandidate] = []
    for candidate in candidates:
        index = next(
            (
                position
                for position, existing in enumerate(unique)
                if candidates_share_unsplit_knowledge(
                    existing,
                    candidate,
                    split_knowledge_ids=split_ids,
                )
                or candidates_are_duplicates(existing, candidate)
            ),
            None,
        )
        if index is None:
            unique.append(candidate)
            continue
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
