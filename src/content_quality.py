from __future__ import annotations

import re
from dataclasses import replace

from .models import KnowledgeItem, KnowledgeReview

WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
CLAUSE_CONJUNCTION_RE = re.compile(
    r"\b(?:и|а|но|and|but|while|whereas)\b",
    flags=re.IGNORECASE | re.UNICODE,
)
META_CONTENT_PATTERNS = (
    re.compile(
        r"\b(?:подпишитесь|ставьте лайк|ссылка в описании|"
        r"в следующем видео|сегодня мы поговорим|спасибо за просмотр)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:subscribe|like and subscribe|link in the description|"
        r"in the next video|today we will discuss|thanks for watching)\b",
        flags=re.IGNORECASE,
    ),
)


def atomicity_issues(
    statement: str,
    *,
    max_words: int = 28,
    max_conjunctions: int = 1,
) -> tuple[str, ...]:
    issues: list[str] = []
    if len(WORD_RE.findall(statement)) > max_words:
        issues.append("too_long")
    if len(CLAUSE_CONJUNCTION_RE.findall(statement)) > max_conjunctions:
        issues.append("multiple_clauses")
    if ";" in statement and "multiple_clauses" not in issues:
        issues.append("multiple_clauses")
    return tuple(issues)


def is_probable_meta_content(item: KnowledgeItem) -> bool:
    text = f"{item.statement} {item.explanation}"
    return any(pattern.search(text) for pattern in META_CONTENT_PATTERNS)


def enforce_content_quality(
    item: KnowledgeItem,
    review: KnowledgeReview,
) -> KnowledgeReview:
    if is_probable_meta_content(item):
        return replace(
            review,
            status="discard",
            confidence=max(review.confidence, 0.98),
            issues=list(dict.fromkeys(review.issues + ["low_learning_value"])),
            recommended_action="discard",
            reason="Мета-контент видео, не являющийся учебным знанием.",
        )

    if item.type in {"comparison", "process"}:
        return review

    issues = atomicity_issues(item.statement)
    if not issues:
        return review
    return replace(
        review,
        status="split",
        confidence=max(review.confidence, 0.95),
        issues=list(dict.fromkeys(review.issues + ["contains_two_facts"])),
        recommended_action="split",
        reason="Автоматическая проверка атомарности: " + ", ".join(issues) + ".",
    )
