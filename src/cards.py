from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    CardCandidate,
    CardCandidateCollection,
    KnowledgeItem,
    KnowledgeReview,
)
from .card_enrichment import (
    CardProposal,
    deduplicate_candidates,
    expression_is_relevant_to_text,
    fallback_card_proposals,
    normalize_front,
    number_candidates,
    proposal_to_candidate,
    localize_card_text,
)
from .llm import LLMError, LLMInvalidResponseError, StructuredLLM, generate_structured, provider_label
from .ollama_schemas import card_response_schema
from .progress import ActivityIndicator
from .review import (
    KnowledgeReviewError,
    load_knowledge_items,
    normalize_review_relations,
)
from .semantic import save_json_atomic
from .target_expressions import load_target_expressions

LOGGER = logging.getLogger("anki_pipeline.cards")

ALLOWED_CARD_TYPES = {
    "basic",
    "reverse",
    "cloze",
    "comparison",
    "process",
    "mistake",
    "definition",
}
CARD_PROPOSAL_FIELDS = {
    "knowledge_ids",
    "card_type",
    "front",
    "back",
    "extra",
    "difficulty",
}
CARD_RESPONSE_SCHEMA = card_response_schema(ALLOWED_CARD_TYPES, CARD_PROPOSAL_FIELDS)
AMBIGUOUS_STARTS = {
    "это",
    "этот",
    "эта",
    "данный",
    "данная",
    "он",
    "она",
    "они",
    "this",
    "it",
    "he",
    "she",
    "they",
}


class CardGenerationError(RuntimeError):
    pass


def load_knowledge_reviews(path: Path) -> tuple[str, str, list[KnowledgeReview]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        reviews = [
            KnowledgeReview(
                knowledge_id=str(item["knowledge_id"]),
                status=str(item["status"]),
                confidence=float(item["confidence"]),
                issues=[str(issue) for issue in item["issues"]],
                recommended_action=str(item["recommended_action"]),
                reason=str(item["reason"]),
            )
            for item in data["reviews"]
        ]
        source = str(data["source"])
        model = str(data["model"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CardGenerationError(f"Не удалось прочитать knowledge_review.json: {exc}") from exc
    return source, model, normalize_review_relations(reviews)


def validate_review_coverage(
    items: list[KnowledgeItem], reviews: list[KnowledgeReview]
) -> None:
    item_ids = [item.id for item in items]
    review_ids = [review.knowledge_id for review in reviews]
    if review_ids != item_ids:
        raise CardGenerationError(
            "knowledge_review.json должен покрывать все knowledge ID ровно один раз."
        )


def group_card_inputs(
    items: list[KnowledgeItem], reviews: list[KnowledgeReview]
) -> list[tuple[list[KnowledgeItem], list[KnowledgeReview]]]:
    review_by_id = {review.knowledge_id: review for review in reviews}
    groups: dict[int, list[KnowledgeItem]] = defaultdict(list)
    for item in items:
        groups[item.semantic_segment_id].append(item)
    return [
        (group, [review_by_id[item.id] for item in group])
        for group in groups.values()
    ]


def build_cards_prompt(
    items: list[KnowledgeItem],
    reviews: list[KnowledgeReview],
    *,
    previous_response: str | None = None,
    errors: list[str] | None = None,
) -> str:
    correction = ""
    if previous_response is not None and errors:
        correction = (
            "\nПРЕДЫДУЩИЙ ОТВЕТ:\n"
            f"{previous_response}\nОШИБКИ:\n- "
            + "\n- ".join(errors)
            + "\nВерни весь исправленный JSON заново."
        )
    payload = [
        {"knowledge": item.to_dict(), "review": review.__dict__}
        for item, review in zip(items, reviews, strict=True)
    ]
    available_ids = {item.id for item in items}
    merge_requirements: list[str] = []
    split_requirements: list[str] = []
    for review in reviews:
        if review.status == "merge":
            related_ids = [
                related
                for related in re.findall(r"k_\d{4}", review.reason)
                if related in available_ids and related != review.knowledge_id
            ]
            if related_ids:
                merge_requirements.append(
                    f"{review.knowledge_id} должен быть в одной карточке с {related_ids[0]}"
                )
            else:
                merge_requirements.append(
                    f"{review.knowledge_id} должен быть в карточке минимум с ещё одним ID"
                )
        if review.status == "split":
            split_requirements.append(
                f"{review.knowledge_id} должен присутствовать в 2–5 разных карточках"
            )
    mandatory = "\n".join(
        [
            "ОБЯЗАТЕЛЬНЫЕ СВЯЗИ:",
            *(f"- {value}." for value in merge_requirements + split_requirements),
            "Если один ID одновременно участвует в split и merge, включи merge-партнёра хотя бы в одну split-карточку.",
        ]
    )
    return f"""Создай качественные текстовые кандидаты будущих Anki-карточек.
Используй только данные knowledge и review. Не добавляй внешние знания.
Для status=discard не создавай карточек.
Для status=split создай 2–5 атомарных карточек с тем же knowledge_id.
Для status=merge объедини совместимые знания этого списка в одну карточку и укажи все IDs.
Для keep создай одну наиболее полезную карточку; лишние reverse/cloze не добавляй.
card_type только {sorted(ALLOWED_CARD_TYPES)}.
ОБЯЗАТЕЛЬНО преобразуй knowledge type в card_type так:
- fact, example, cause_effect -> basic;
- pitfall -> mistake;
- definition -> definition;
- comparison -> comparison;
- process -> process.
card_type example и fact НЕ СУЩЕСТВУЮТ и запрещены.
front — один самостоятельный вопрос до 120 символов без ответа внутри.
Не начинай front с неоднозначных местоимений вроде "это", "данный", "он", "this", "it".
back — один самостоятельный ответ до 500 символов.
extra — только дополнительный контекст или пример, не основной ответ.
difficulty — целое число 1–5. Не возвращай importance, tags, source IDs или card_id.
Только обычный текст: без HTML, Markdown, CSS, LaTeX, изображений и аудио.
Каждый knowledge ID со статусом не discard должен использоваться; discard — ни разу.
{mandatory}
Верни только строгий JSON без дополнительных полей:
{{"card_candidates":[{{"knowledge_ids":["k_0001"],"card_type":"basic","front":"...","back":"...","extra":"","difficulty":3}}]}}{correction}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}"""


def has_forbidden_formatting(text: str) -> bool:
    return bool(
        re.search(r"<[^>]+>|\*\*|`|\$\$|^\s*#", text, flags=re.MULTILINE)
    )


def parse_card_response(
    data: dict[str, Any],
    items: list[KnowledgeItem],
    reviews: list[KnowledgeReview],
    *,
    raw_response: str = "",
) -> list[CardProposal]:
    errors: list[str] = []
    if set(data) != {"card_candidates"}:
        errors.append("Корневой объект должен содержать только card_candidates.")
    raw_cards = data.get("card_candidates")
    if not isinstance(raw_cards, list):
        raise CardGenerationError("Поле card_candidates должно быть массивом.")
    if "<think>" in raw_response.casefold():
        errors.append("Ответ содержит <think>.")
    item_ids = {item.id for item in items}
    review_by_id = {review.knowledge_id: review for review in reviews}
    discarded = {
        review.knowledge_id for review in reviews if review.status == "discard"
    }
    proposals: list[CardProposal] = []
    fronts: set[str] = set()
    for index, raw in enumerate(raw_cards):
        prefix = f"Card {index + 1}: "
        if not isinstance(raw, dict):
            errors.append(prefix + "значение должно быть объектом.")
            continue
        if set(raw) != CARD_PROPOSAL_FIELDS:
            errors.append(prefix + "набор полей не соответствует схеме.")
        knowledge_ids = raw.get("knowledge_ids")
        card_type = raw.get("card_type")
        front = raw.get("front")
        back = raw.get("back")
        extra = raw.get("extra")
        difficulty = raw.get("difficulty")
        valid_ids = (
            isinstance(knowledge_ids, list)
            and bool(knowledge_ids)
            and all(isinstance(item, str) for item in knowledge_ids)
            and len(knowledge_ids) == len(set(knowledge_ids))
        )
        if not valid_ids:
            errors.append(prefix + "knowledge_ids должен быть непустым уникальным списком.")
        else:
            unknown = [item for item in knowledge_ids if item not in item_ids]
            forbidden = [item for item in knowledge_ids if item in discarded]
            if unknown:
                errors.append(prefix + f"неизвестные knowledge IDs: {unknown}.")
            if forbidden:
                errors.append(prefix + f"discard IDs запрещены: {forbidden}.")
        if card_type not in ALLOWED_CARD_TYPES:
            errors.append(prefix + f"неизвестный card_type {card_type!r}.")
        if not isinstance(front, str) or not front.strip():
            errors.append(prefix + "front пустой.")
        else:
            front = front.strip()
            if len(front) > 120:
                errors.append(prefix + "front длиннее 120 символов.")
            if front.count("?") > 1:
                errors.append(prefix + "front содержит несколько вопросов.")
            first_word = normalize_front(front).split(maxsplit=1)[0].strip(".,?!:;")
            if first_word in AMBIGUOUS_STARTS:
                errors.append(prefix + "front начинается с неоднозначного местоимения.")
            normalized = normalize_front(front)
            if normalized in fronts:
                errors.append(prefix + "дублирующийся front.")
            fronts.add(normalized)
        if not isinstance(back, str) or not back.strip():
            errors.append(prefix + "back пустой.")
        else:
            back = back.strip()
            if len(back) > 500:
                errors.append(prefix + "back длиннее 500 символов.")
            if isinstance(front, str) and normalize_front(back) in normalize_front(front):
                errors.append(prefix + "front содержит ответ.")
        if not isinstance(extra, str):
            errors.append(prefix + "extra должен быть строкой.")
        if any(
            isinstance(value, str) and has_forbidden_formatting(value)
            for value in (front, back, extra)
        ):
            errors.append(prefix + "обнаружено запрещённое форматирование.")
        if (
            not isinstance(difficulty, int)
            or isinstance(difficulty, bool)
            or not 1 <= difficulty <= 5
        ):
            errors.append(prefix + "difficulty должен быть целым числом 1–5.")
        if not any(error.startswith(prefix) for error in errors):
            proposals.append(
                CardProposal(
                    knowledge_ids=knowledge_ids,
                    card_type=card_type,
                    front=front,
                    back=back,
                    extra=extra.strip(),
                    difficulty=difficulty,
                )
            )

    usage = Counter(
        knowledge_id for proposal in proposals for knowledge_id in proposal.knowledge_ids
    )
    for review in reviews:
        count = usage[review.knowledge_id]
        if review.status == "discard" and count:
            errors.append(f"Discard ID {review.knowledge_id} использован в карточке.")
        elif review.status == "split" and not 2 <= count <= 5:
            errors.append(
                f"Split ID {review.knowledge_id} должен дать 2–5 карточек, получено {count}."
            )
        elif review.status == "merge":
            merged = any(
                review.knowledge_id in proposal.knowledge_ids
                and len(proposal.knowledge_ids) > 1
                for proposal in proposals
            )
            if not merged:
                errors.append(f"Merge ID {review.knowledge_id} не объединён.")
        elif review.status == "keep" and count < 1:
            errors.append(f"Keep ID {review.knowledge_id} не использован.")
    if errors:
        raise CardGenerationError(" | ".join(errors))
    return proposals


def save_cards_diagnostic(
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


def generate_card_candidates(
    knowledge_path: Path,
    review_path: Path,
    destination: Path,
    diagnostic_path: Path,
    client: StructuredLLM,
    expressions_path: Path | None = None,
) -> CardCandidateCollection:
    source, _, items = load_knowledge_items(knowledge_path)
    _, _, reviews = load_knowledge_reviews(review_path)
    validate_review_coverage(items, reviews)
    groups = group_card_inputs(items, reviews)
    proposals: list[CardProposal] = []
    had_fallback = False
    for group_number, (group_items, group_reviews) in enumerate(groups, start=1):
        LOGGER.info("Card Candidates: группа %d/%d", group_number, len(groups))
        responses: list[str] = []
        previous: str | None = None
        validation_errors: list[str] | None = None
        for attempt in range(2):
            label = f"{provider_label(client)} Cards: группа {group_number}/{len(groups)}"
            if attempt:
                label += ", повторная попытка"
            try:
                prompt = build_cards_prompt(
                    group_items,
                    group_reviews,
                    previous_response=previous,
                    errors=validation_errors,
                )
                with ActivityIndicator(label):
                    data, raw = generate_structured(client, prompt, schema=CARD_RESPONSE_SCHEMA)
                responses.append(raw)
                proposals.extend(
                    parse_card_response(
                        data, group_items, group_reviews, raw_response=raw
                    )
                )
                break
            except LLMInvalidResponseError as exc:
                previous = exc.raw_response
                responses.append(exc.raw_response)
                validation_errors = [str(exc)]
            except CardGenerationError as exc:
                previous = responses[-1] if responses else ""
                validation_errors = str(exc).split(" | ")
            except LLMError:
                raise
            if attempt == 1:
                message = " | ".join(validation_errors or ["Неизвестная ошибка."])
                ids = [item.id for item in group_items]
                save_cards_diagnostic(
                    diagnostic_path,
                    knowledge_ids=ids,
                    error_message=message,
                    responses=responses,
                    model=client.model,
                )
                LOGGER.warning(
                    "Card Candidates группы %d заменены безопасными черновиками после "
                    "двух некорректных ответов. Диагностика: %s",
                    group_number,
                    diagnostic_path,
                )
                proposals.extend(fallback_card_proposals(group_items, group_reviews))
                had_fallback = True
                break
    knowledge_by_id = {item.id: item for item in items}
    expressions_by_knowledge: dict[str, list] = defaultdict(list)
    verified_expressions: list = []
    if expressions_path and expressions_path.exists():
        for expression in load_target_expressions(expressions_path):
            if expression.verification not in {"verified", "transcribed"}:
                continue
            verified_expressions.append(expression)
            for knowledge_id in expression.knowledge_ids:
                expressions_by_knowledge[knowledge_id].append(expression)
    candidates = [
        proposal_to_candidate(
            item, knowledge_by_id, expressions_by_knowledge, verified_expressions
        )
        for item in proposals
    ]
    candidates = deduplicate_candidates(candidates)
    candidates = number_candidates(candidates)
    result = CardCandidateCollection(source, client.model, candidates)
    save_json_atomic(result.to_dict(), destination)
    if not had_fallback:
        diagnostic_path.unlink(missing_ok=True)
    return result


def localize_existing_candidates(cards_path: Path, expressions_path: Path) -> int:
    try:
        data = json.loads(cards_path.read_text(encoding="utf-8"))
        candidates = [CardCandidate(**item) for item in data["card_candidates"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CardGenerationError(f"Не удалось прочитать card_candidates.json: {exc}") from exc
    verified = [
        item for item in load_target_expressions(expressions_path)
        if item.verification in {"verified", "transcribed"}
    ]
    by_id = {item.id: item for item in verified}
    localized: list[CardCandidate] = []
    for candidate in candidates:
        source_ids = set(candidate.source_transcript_segment_ids)
        linked_by_id = {
            item.id: item
            for item in verified
            if (
                item.id in (candidate.expression_ids or [])
                or bool(source_ids & set(item.source_transcript_segment_ids))
            )
            and expression_is_relevant_to_text(
                item, " ".join((candidate.front, candidate.back))
            )
        }
        linked = list(linked_by_id.values())
        front, back, extra = localize_card_text(
            candidate.front, candidate.back, candidate.extra, linked, verified
        )
        localized.append(replace(
            candidate,
            front=front,
            back=back,
            extra=extra,
            expression_ids=list(linked_by_id),
        ))
    save_json_atomic(
        {**data, "card_candidates": [item.__dict__ for item in localized]}, cards_path
    )
    return len(localized)
