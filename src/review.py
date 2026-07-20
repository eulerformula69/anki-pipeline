from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    DuplicateOccurrence,
    KnowledgeItem,
    KnowledgeReview,
    KnowledgeReviewCollection,
)
from .ollama_client import OllamaClient, OllamaError, OllamaInvalidResponseError
from .ollama_schemas import review_response_schema
from .progress import ActivityIndicator
from .semantic import save_json_atomic

LOGGER = logging.getLogger("anki_pipeline.review")

ALLOWED_REVIEW_STATUSES = {"keep", "merge", "split", "discard"}
ALLOWED_REVIEW_ISSUES = {
    "too_general",
    "too_specific",
    "contains_two_facts",
    "duplicate",
    "weak_example",
    "redundant",
    "unclear",
    "low_learning_value",
}
ALLOWED_RECOMMENDED_ACTIONS = {"none", "rewrite", "split", "merge", "discard"}
REVIEW_FIELDS = {
    "knowledge_id",
    "status",
    "confidence",
    "issues",
    "recommended_action",
    "reason",
}
REVIEW_RESPONSE_SCHEMA = review_response_schema(
    ALLOWED_REVIEW_STATUSES,
    ALLOWED_REVIEW_ISSUES,
    ALLOWED_RECOMMENDED_ACTIONS,
    REVIEW_FIELDS,
)


class KnowledgeReviewError(RuntimeError):
    pass


def normalize_review_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if 0 <= value <= 1:
        return float(value)
    if isinstance(value, int) and 2 <= value <= 5:
        return value / 5
    return None


def normalize_review_relations(
    reviews: list[KnowledgeReview],
) -> list[KnowledgeReview]:
    by_id = {review.knowledge_id: review for review in reviews}
    normalized: list[KnowledgeReview] = []
    for review in reviews:
        if review.status != "merge":
            normalized.append(review)
            continue
        targets = [
            item_id
            for item_id in re.findall(r"k_\d{4}", review.reason)
            if item_id != review.knowledge_id and item_id in by_id
        ]
        if targets and all(by_id[item_id].status == "discard" for item_id in targets):
            normalized.append(replace(
                review,
                status="keep",
                recommended_action="none",
                issues=[
                    issue for issue in review.issues
                    if issue not in {"duplicate", "redundant"}
                ],
                reason=(
                    "Основное знание сохранено; дубли "
                    f"{', '.join(targets)} отброшены."
                ),
            ))
        else:
            normalized.append(review)
    return normalized


def load_knowledge_items(path: Path) -> tuple[str, str, list[KnowledgeItem]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [
            KnowledgeItem(
                id=str(item["id"]),
                semantic_segment_id=int(item["semantic_segment_id"]),
                type=str(item["type"]),
                statement=str(item["statement"]),
                explanation=str(item["explanation"]),
                importance=int(item["importance"]),
                source_transcript_segment_ids=[
                    int(segment_id)
                    for segment_id in item["source_transcript_segment_ids"]
                ],
                evidence=str(item["evidence"]),
                duplicate_occurrences=(
                    [
                        DuplicateOccurrence(
                            semantic_segment_id=int(occurrence["semantic_segment_id"]),
                            source_transcript_segment_ids=[
                                int(segment_id)
                                for segment_id in occurrence[
                                    "source_transcript_segment_ids"
                                ]
                            ],
                        )
                        for occurrence in item["duplicate_occurrences"]
                    ]
                    if item.get("duplicate_occurrences")
                    else None
                ),
            )
            for item in data["knowledge_items"]
        ]
        source = str(data["source"])
        model = str(data["model"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise KnowledgeReviewError(f"Не удалось прочитать knowledge_items.json: {exc}") from exc
    if not items:
        raise KnowledgeReviewError("knowledge_items.json не содержит знаний.")
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise KnowledgeReviewError("knowledge_items.json содержит дублирующиеся ID.")
    return source, model, items


def group_knowledge_items(
    items: list[KnowledgeItem], *, max_group_size: int = 12
) -> list[list[KnowledgeItem]]:
    if max_group_size < 1:
        raise ValueError("max_group_size должен быть положительным.")
    return [
        items[start:start + max_group_size]
        for start in range(0, len(items), max_group_size)
    ]


def build_review_prompt(
    items: list[KnowledgeItem],
    *,
    previous_response: str | None = None,
    errors: list[str] | None = None,
) -> str:
    payload = [item.to_dict() for item in items]
    correction = ""
    if previous_response is not None and errors:
        correction = (
            "\nПРЕДЫДУЩИЙ ОТВЕТ:\n"
            f"{previous_response}\nОШИБКИ:\n- "
            + "\n- ".join(errors)
            + "\nВерни весь исправленный JSON заново."
        )
    return f"""Ты редактор качества учебных знаний, а не генератор новых знаний.
Оцени каждый входной knowledge item, ничего в нём не изменяя.
Найди слишком общие, частные, длинные, неясные, составные, дублирующиеся,
слабые примеры и знания с низкой учебной ценностью.
Оцени весь список глобально, включая знания из разных semantic_segment_id.
Итоговый набор должен быть компактным: одна сильная учебная цель вместо нескольких
переформулировок одного правила. Для смысловых повторов используй merge или discard.
Не сохраняй отдельную карточку-пример, если тот же пример лучше поместить в explanation
основного правила и он не проверяет самостоятельный навык.
status: keep, merge, split или discard.
issues: только {sorted(ALLOWED_REVIEW_ISSUES)}.
recommended_action: только {sorted(ALLOWED_RECOMMENDED_ACTIONS)}.
confidence — число 0..1. reason — краткое объяснение на языке knowledge item.
Для merge укажи в reason ID совместимого знания из этого же списка.
Верни каждый входной knowledge_id ровно один раз и в исходном порядке.
Не возвращай Markdown, исправленные statements или дополнительные поля.
Строгий JSON:
{{"reviews":[{{"knowledge_id":"k_0001","status":"keep","confidence":0.94,"issues":[],"recommended_action":"none","reason":"..."}}]}}{correction}

KNOWLEDGE ITEMS:
{json.dumps(payload, ensure_ascii=False, indent=2)}"""


def parse_review_response(
    data: dict[str, Any], expected_ids: list[str], *, raw_response: str = ""
) -> list[KnowledgeReview]:
    errors: list[str] = []
    if set(data) != {"reviews"}:
        errors.append("Корневой объект должен содержать только reviews.")
    raw_reviews = data.get("reviews")
    if not isinstance(raw_reviews, list):
        raise KnowledgeReviewError("Поле reviews должно быть массивом.")
    if "<think>" in raw_response.casefold():
        errors.append("Ответ содержит <think>.")
    reviews: list[KnowledgeReview] = []
    for index, raw in enumerate(raw_reviews):
        prefix = f"Review {index + 1}: "
        if not isinstance(raw, dict):
            errors.append(prefix + "значение должно быть объектом.")
            continue
        if set(raw) != REVIEW_FIELDS:
            errors.append(prefix + "набор полей не соответствует схеме.")
        knowledge_id = raw.get("knowledge_id")
        status = raw.get("status")
        confidence = normalize_review_confidence(raw.get("confidence"))
        issues = raw.get("issues")
        action = raw.get("recommended_action")
        reason = raw.get("reason")
        if not isinstance(knowledge_id, str):
            errors.append(prefix + "knowledge_id должен быть строкой.")
        if status not in ALLOWED_REVIEW_STATUSES:
            errors.append(prefix + f"неизвестный status {status!r}.")
        if confidence is None:
            errors.append(
                prefix
                + "confidence должен быть числом от 0 до 1 или целым числом от 1 до 5."
            )
        if not isinstance(issues, list):
            errors.append(prefix + "issues содержит неизвестные значения.")
        elif any(
            not isinstance(issue, str) or issue not in ALLOWED_REVIEW_ISSUES
            for issue in issues
        ):
            errors.append(prefix + "issues содержит неизвестные значения.")
        if action not in ALLOWED_RECOMMENDED_ACTIONS:
            errors.append(prefix + f"неизвестный recommended_action {action!r}.")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(prefix + "reason должен быть непустой строкой.")
        if not any(error.startswith(prefix) for error in errors):
            reviews.append(
                KnowledgeReview(
                    knowledge_id=knowledge_id,
                    status=status,
                    confidence=float(confidence),
                    issues=issues,
                    recommended_action=action,
                    reason=reason.strip(),
                )
            )
    actual_ids = [review.knowledge_id for review in reviews]
    if actual_ids != expected_ids:
        errors.append(
            f"Нарушено покрытие knowledge ID: ожидалось {expected_ids}, получено {actual_ids}."
        )
    if errors:
        raise KnowledgeReviewError(" | ".join(errors))
    return normalize_review_relations(reviews)


def save_review_diagnostic(
    path: Path,
    *,
    knowledge_ids: list[str],
    error_message: str,
    responses: list[str],
    model: str,
) -> None:
    save_json_atomic(
        {
            "knowledge_ids": knowledge_ids,
            "errors": error_message,
            "ollama_responses": responses,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def review_knowledge(
    knowledge_path: Path,
    destination: Path,
    diagnostic_path: Path,
    client: OllamaClient,
) -> KnowledgeReviewCollection:
    if isinstance(client, OllamaClient):
        client.format_schema = REVIEW_RESPONSE_SCHEMA
    source, _, items = load_knowledge_items(knowledge_path)
    groups = group_knowledge_items(items)
    reviews: list[KnowledgeReview] = []
    for group_number, group in enumerate(groups, start=1):
        LOGGER.info("Knowledge Review: группа %d/%d", group_number, len(groups))
        expected_ids = [item.id for item in group]
        responses: list[str] = []
        previous: str | None = None
        validation_errors: list[str] | None = None
        for attempt in range(2):
            label = f"Ollama Review: группа {group_number}/{len(groups)}"
            if attempt:
                label += ", повторная попытка"
            try:
                prompt = build_review_prompt(
                    group, previous_response=previous, errors=validation_errors
                )
                with ActivityIndicator(label):
                    data, raw = client.generate_json(prompt)
                responses.append(raw)
                reviews.extend(
                    parse_review_response(data, expected_ids, raw_response=raw)
                )
                break
            except OllamaInvalidResponseError as exc:
                previous = exc.raw_response
                responses.append(exc.raw_response)
                validation_errors = [str(exc)]
            except KnowledgeReviewError as exc:
                previous = responses[-1] if responses else ""
                validation_errors = str(exc).split(" | ")
            except OllamaError:
                raise
            if attempt == 1:
                message = " | ".join(validation_errors or ["Неизвестная ошибка."])
                save_review_diagnostic(
                    diagnostic_path,
                    knowledge_ids=expected_ids,
                    error_message=message,
                    responses=responses,
                    model=client.model,
                )
                raise KnowledgeReviewError(
                    "Ollama дважды вернул некорректный Knowledge Review. "
                    f"Диагностика: {diagnostic_path}"
                )
    result = KnowledgeReviewCollection(source, client.model, reviews)
    save_json_atomic(result.to_dict(), destination)
    diagnostic_path.unlink(missing_ok=True)
    return result
