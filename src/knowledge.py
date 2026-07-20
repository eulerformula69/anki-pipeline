from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    DuplicateOccurrence,
    KnowledgeCollection,
    KnowledgeItem,
    SemanticSegment,
    TranscriptSegment,
)
from .llm import (
    LLMError, LLMInvalidResponseError, StructuredLLM,
    generate_structured, provider_label,
)
from .progress import ActivityIndicator
from .semantic import save_json_atomic

LOGGER = logging.getLogger("anki_pipeline.knowledge")

ALLOWED_TYPES = {
    "definition",
    "fact",
    "cause_effect",
    "comparison",
    "process",
    "example",
    "pitfall",
}
DEDUP_STOPWORDS = {
    "a", "an", "and", "as", "for", "in", "is", "of", "or", "that", "the", "to",
    "это", "и", "в", "на", "что", "как", "для", "из", "с", "по",
}
PROPOSAL_FIELDS = {
    "type",
    "statement",
    "explanation",
    "importance",
    "source_transcript_segment_ids",
}


def knowledge_response_schema(max_items: int) -> dict[str, Any]:
    return {
        "type": "object", "required": ["knowledge_items"],
        "properties": {"knowledge_items": {
            "type": "array", "maxItems": max_items,
            "items": {
                "type": "object",
                "required": sorted(PROPOSAL_FIELDS),
                "properties": {
                    "type": {"type": "string", "enum": sorted(ALLOWED_TYPES)},
                    "statement": {"type": "string"},
                    "explanation": {"type": "string"},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "source_transcript_segment_ids": {
                        "type": "array", "minItems": 1,
                        "items": {"type": "integer"},
                    },
                },
            },
        }},
    }


class KnowledgeExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class KnowledgeProposal:
    semantic_segment_id: int
    type: str
    statement: str
    explanation: str
    importance: int
    source_transcript_segment_ids: list[int]


def load_knowledge_sources(
    transcript_path: Path, semantic_path: Path
) -> tuple[str, str, list[TranscriptSegment], list[SemanticSegment]]:
    try:
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
        semantic_data = json.loads(semantic_path.read_text(encoding="utf-8"))
        transcript_segments = [
            TranscriptSegment(
                id=int(item["id"]),
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
            )
            for item in transcript_data["segments"]
        ]
        semantic_segments = [
            SemanticSegment(
                id=int(item["id"]),
                title=str(item["title"]),
                start=float(item["start"]),
                end=float(item["end"]),
                summary=str(item["summary"]),
                transcript_segment_ids=[
                    int(segment_id) for segment_id in item["transcript_segment_ids"]
                ],
                text=str(item["text"]),
            )
            for item in semantic_data["segments"]
        ]
        source = str(transcript_data["source"])
        language = str(transcript_data["language"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise KnowledgeExtractionError(f"Не удалось прочитать исходные JSON: {exc}") from exc

    if not transcript_segments:
        raise KnowledgeExtractionError("transcript.json не содержит сегментов.")
    if not semantic_segments:
        raise KnowledgeExtractionError("semantic_segments.json не содержит блоков.")
    transcript_ids = {segment.id for segment in transcript_segments}
    for segment in semantic_segments:
        if not segment.transcript_segment_ids:
            raise KnowledgeExtractionError(
                f"Смысловой блок {segment.id} не содержит исходных ID."
            )
        if not set(segment.transcript_segment_ids) <= transcript_ids:
            raise KnowledgeExtractionError(
                f"Смысловой блок {segment.id} ссылается на отсутствующие transcript ID."
            )
    return source, language, transcript_segments, semantic_segments


def build_knowledge_prompt(
    semantic: SemanticSegment,
    transcript_by_id: dict[int, TranscriptSegment],
    *,
    result_language: str,
    max_items: int,
    previous_response: str | None = None,
    errors: list[str] | None = None,
) -> str:
    fragments = "\n".join(
        f"{segment_id}: {transcript_by_id[segment_id].text}"
        for segment_id in semantic.transcript_segment_ids
    )
    correction = ""
    if previous_response is not None and errors:
        correction = (
            "\nПРЕДЫДУЩИЙ ОТВЕТ:\n"
            f"{previous_response}\n"
            "ОШИБКИ:\n- "
            + "\n- ".join(errors)
            + "\nВерни весь исправленный JSON заново."
        )
    language_instruction = {
        "ru": "Пиши statement и explanation на русском языке.",
        "en": "Write statement and explanation in English.",
    }.get(
        result_language,
        "Пиши statement и explanation на языке исходной транскрипции: "
        f"{result_language}.",
    )
    return f"""Извлеки атомарные проверяемые учебные знания только из данного блока.
Не используй внешние знания и не добавляй отсутствующие сведения.
Исключи приветствия, рекламу, переходы, повторы, пустые вводные и неучебные мнения.
Каждый item должен содержать ровно один тезис, не вопрос и не будущую Anki-карточку.
Даже для type=example statement должен быть утверждением и оканчиваться точкой;
если пример в источнике является вопросом, опиши его как "An example is the phrase ...".
Не возвращай один и тот же тезис повторно под другим type или в другой формулировке.
Выбирай минимальный достаточный набор source IDs и указывай их по возрастанию.
Разрешённые type: {sorted(ALLOWED_TYPES)}.
importance — целое число 1–5 относительно этого блока.
Не возвращай evidence и id: программа сформирует их из оригинала.
Не переводи и не искажай имена, формулы, обозначения и специальные термины.
{language_instruction}
Верни только JSON строго этой формы без дополнительных полей:
{{"knowledge_items":[{{"type":"fact","statement":"...","explanation":"","importance":3,"source_transcript_segment_ids":[1]}}]}}
Допустим пустой массив. Максимум items: {max_items}.{correction}

Смысловой блок {semantic.id}
Title: {semantic.title}
Summary: {semantic.summary}
Text: {semantic.text}
Target-language text: {semantic.target_language_text}
Исходные фрагменты:
{fragments}"""


def parse_knowledge_response(
    data: dict[str, Any],
    semantic: SemanticSegment,
    transcript_by_id: dict[int, TranscriptSegment],
    *,
    max_items: int,
    raw_response: str = "",
) -> list[KnowledgeProposal]:
    errors: list[str] = []
    if set(data) != {"knowledge_items"}:
        errors.append("Корневой объект должен содержать только knowledge_items.")
    raw_items = data.get("knowledge_items")
    if not isinstance(raw_items, list):
        raise KnowledgeExtractionError("Поле knowledge_items должно быть массивом.")
    if len(raw_items) > max_items:
        errors.append(f"Превышен лимит {max_items} items на смысловой блок.")
    if "<think>" in raw_response.casefold() or "</think>" in raw_response.casefold():
        errors.append("Ответ содержит внутренние рассуждения <think>.")

    allowed_ids = semantic.transcript_segment_ids
    allowed_set = set(allowed_ids)
    proposals: list[KnowledgeProposal] = []
    for index, raw in enumerate(raw_items):
        prefix = f"Item {index + 1}: "
        if not isinstance(raw, dict):
            errors.append(prefix + "значение должно быть объектом.")
            continue
        if set(raw) != PROPOSAL_FIELDS:
            errors.append(prefix + "набор полей не соответствует схеме.")
        item_type = raw.get("type")
        statement = raw.get("statement")
        explanation = raw.get("explanation")
        importance = raw.get("importance")
        source_ids = raw.get("source_transcript_segment_ids")
        if item_type not in ALLOWED_TYPES:
            errors.append(prefix + f"неизвестный type {item_type!r}.")
        if not isinstance(statement, str) or not statement.strip():
            errors.append(prefix + "statement пустой.")
        elif statement.strip().endswith(("?", "？")):
            errors.append(prefix + "statement не должен быть вопросом.")
        if not isinstance(explanation, str):
            errors.append(prefix + "explanation должен быть строкой.")
        if (
            not isinstance(importance, int)
            or isinstance(importance, bool)
            or not 1 <= importance <= 5
        ):
            errors.append(prefix + "importance должен быть целым числом от 1 до 5.")
        valid_ids = (
            isinstance(source_ids, list)
            and bool(source_ids)
            and all(isinstance(item, int) and not isinstance(item, bool) for item in source_ids)
        )
        if not valid_ids:
            errors.append(prefix + "source IDs должны быть непустым массивом целых чисел.")
        else:
            normalized_source_ids = sorted(set(source_ids))
            if source_ids != normalized_source_ids:
                LOGGER.warning(
                    "Item %d: source IDs нормализованы: %s -> %s",
                    index + 1,
                    source_ids,
                    normalized_source_ids,
                )
            source_ids = normalized_source_ids
            missing = [item for item in source_ids if item not in transcript_by_id]
            outside = [item for item in source_ids if item not in allowed_set]
            if missing:
                errors.append(prefix + f"transcript ID не существуют: {missing}.")
            if outside:
                errors.append(prefix + f"ID не принадлежат смысловому блоку: {outside}.")
        if not errors or not any(error.startswith(prefix) for error in errors):
            proposals.append(
                KnowledgeProposal(
                    semantic_segment_id=semantic.id,
                    type=item_type,
                    statement=statement.strip(),
                    explanation=explanation.strip(),
                    importance=importance,
                    source_transcript_segment_ids=source_ids,
                )
            )
    if errors and proposals and all(error.startswith("Item ") for error in errors):
        LOGGER.warning(
            "Пропущено некорректных знаний: %d; корректные сохранены.",
            len(errors),
        )
        return proposals
    if errors:
        raise KnowledgeExtractionError(" | ".join(errors))
    return proposals


def normalize_evidence(text: str) -> str:
    return " ".join(text.split()).casefold()


def build_evidence(
    source_ids: list[int], transcript_by_id: dict[int, TranscriptSegment]
) -> str:
    return " ".join(transcript_by_id[item].text.strip() for item in source_ids).strip()


def evidence_matches(
    evidence: str,
    source_ids: list[int],
    transcript_by_id: dict[int, TranscriptSegment],
) -> bool:
    return normalize_evidence(evidence) == normalize_evidence(
        build_evidence(source_ids, transcript_by_id)
    )


def proposal_to_item(
    proposal: KnowledgeProposal,
    transcript_by_id: dict[int, TranscriptSegment],
) -> KnowledgeItem:
    evidence = build_evidence(proposal.source_transcript_segment_ids, transcript_by_id)
    if not evidence or not evidence_matches(
        evidence, proposal.source_transcript_segment_ids, transcript_by_id
    ):
        raise KnowledgeExtractionError("Evidence не совпадает с оригинальной транскрипцией.")
    return KnowledgeItem(
        id="",
        semantic_segment_id=proposal.semantic_segment_id,
        type=proposal.type,
        statement=proposal.statement,
        explanation=proposal.explanation,
        importance=proposal.importance,
        source_transcript_segment_ids=proposal.source_transcript_segment_ids,
        evidence=evidence,
    )


def normalize_statement(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()


def statements_are_obvious_duplicates(
    first: KnowledgeItem, second: KnowledgeItem
) -> bool:
    first_normalized = normalize_statement(first.statement)
    second_normalized = normalize_statement(second.statement)
    if first_normalized == second_normalized:
        return True
    if first.type == second.type and SequenceMatcher(
        None, first_normalized, second_normalized
    ).ratio() >= 0.92:
        return True
    if (
        first.semantic_segment_id != second.semantic_segment_id
        or first.source_transcript_segment_ids != second.source_transcript_segment_ids
    ):
        return False
    first_tokens = set(first_normalized.split()) - DEDUP_STOPWORDS
    second_tokens = set(second_normalized.split()) - DEDUP_STOPWORDS
    shared = first_tokens & second_tokens
    union = first_tokens | second_tokens
    return len(shared) >= 4 and bool(union) and len(shared) / len(union) >= 0.3


def deduplicate_items(
    items: list[KnowledgeItem], transcript_by_id: dict[int, TranscriptSegment]
) -> list[KnowledgeItem]:
    unique: list[KnowledgeItem] = []
    for item in items:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(unique)
                if statements_are_obvious_duplicates(existing, item)
            ),
            None,
        )
        if duplicate_index is None:
            unique.append(item)
            continue
        existing = unique[duplicate_index]
        if existing.semantic_segment_id == item.semantic_segment_id:
            combined_ids = sorted(
                set(existing.source_transcript_segment_ids)
                | set(item.source_transcript_segment_ids)
            )
            unique[duplicate_index] = replace(
                existing,
                importance=max(existing.importance, item.importance),
                source_transcript_segment_ids=combined_ids,
                evidence=build_evidence(combined_ids, transcript_by_id),
            )
        else:
            occurrences = list(existing.duplicate_occurrences or [])
            occurrences.append(
                DuplicateOccurrence(
                    semantic_segment_id=item.semantic_segment_id,
                    source_transcript_segment_ids=item.source_transcript_segment_ids,
                )
            )
            unique[duplicate_index] = replace(
                existing, duplicate_occurrences=occurrences
            )
    return unique


def save_knowledge_diagnostic(
    path: Path,
    *,
    semantic_segment_id: int,
    error_message: str,
    responses: list[str],
    model: str,
) -> None:
    save_json_atomic(
        {
            "semantic_segment_id": semantic_segment_id,
            "errors": error_message,
            "ollama_responses": responses,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def extract_knowledge(
    transcript_path: Path,
    semantic_path: Path,
    destination: Path,
    diagnostic_path: Path,
    client: StructuredLLM,
    *,
    language: str = "source",
    min_importance: int = 1,
    max_items_per_segment: int = 12,
) -> KnowledgeCollection:
    if language not in {"source", "ru", "en"}:
        raise KnowledgeExtractionError("knowledge language должен быть source, ru или en.")
    if not 1 <= min_importance <= 5:
        raise KnowledgeExtractionError("knowledge-min-importance должен быть от 1 до 5.")
    if max_items_per_segment < 1:
        raise KnowledgeExtractionError("knowledge-max-items-per-segment должен быть не меньше 1.")

    source, source_language, transcript, semantic_segments = load_knowledge_sources(
        transcript_path, semantic_path
    )
    transcript_by_id = {segment.id: segment for segment in transcript}
    result_language = source_language if language == "source" else language
    proposals: list[KnowledgeProposal] = []
    had_fallback = False

    for block_number, semantic in enumerate(semantic_segments, start=1):
        LOGGER.info(
            "Извлечение знаний: блок %d/%d", block_number, len(semantic_segments)
        )
        raw_responses: list[str] = []
        previous_response: str | None = None
        validation_errors: list[str] | None = None
        for attempt in range(2):
            label = f"{provider_label(client)}: блок {block_number}/{len(semantic_segments)}"
            if attempt:
                label += ", повторная попытка"
            try:
                prompt = build_knowledge_prompt(
                    semantic,
                    transcript_by_id,
                    result_language=result_language,
                    max_items=max_items_per_segment,
                    previous_response=previous_response,
                    errors=validation_errors,
                )
                with ActivityIndicator(label):
                    data, raw = generate_structured(
                        client, prompt,
                        schema=knowledge_response_schema(max_items_per_segment),
                    )
                raw_responses.append(raw)
                block_proposals = parse_knowledge_response(
                    data,
                    semantic,
                    transcript_by_id,
                    max_items=max_items_per_segment,
                    raw_response=raw,
                )
                proposals.extend(block_proposals)
                break
            except LLMInvalidResponseError as exc:
                previous_response = exc.raw_response
                raw_responses.append(exc.raw_response)
                validation_errors = [str(exc)]
            except KnowledgeExtractionError as exc:
                previous_response = raw_responses[-1] if raw_responses else ""
                validation_errors = str(exc).split(" | ")
            except LLMError:
                raise
            if attempt == 1:
                message = " | ".join(validation_errors or ["Неизвестная ошибка ответа."])
                save_knowledge_diagnostic(
                    diagnostic_path,
                    semantic_segment_id=semantic.id,
                    error_message=message,
                    responses=raw_responses,
                    model=client.model,
                )
                LOGGER.warning(
                    "Блок %s пропущен после двух некорректных ответов %s. "
                    "Диагностика: %s",
                    semantic.id,
                    provider_label(client),
                    diagnostic_path,
                )
                proposals.append(KnowledgeProposal(
                    semantic_segment_id=semantic.id,
                    type="fact",
                    statement=(semantic.summary or semantic.title).strip(),
                    explanation="Automatic fallback: manually verify this knowledge.",
                    importance=max(min_importance, 2),
                    source_transcript_segment_ids=list(semantic.transcript_segment_ids),
                ))
                had_fallback = True
                break

    items = [proposal_to_item(proposal, transcript_by_id) for proposal in proposals]
    items = [item for item in items if item.importance >= min_importance]
    items = deduplicate_items(items, transcript_by_id)
    numbered = [replace(item, id=f"k_{index:04d}") for index, item in enumerate(items, 1)]
    result = KnowledgeCollection(source=source, model=client.model, knowledge_items=numbered)
    save_json_atomic(result.to_dict(), destination)
    if not had_fallback:
        diagnostic_path.unlink(missing_ok=True)
    return result
