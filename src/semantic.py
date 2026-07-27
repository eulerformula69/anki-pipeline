from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SemanticSegment, SemanticTranscript, TranscriptSegment
from .llm import (
    LLMError, LLMInvalidResponseError, StructuredLLM,
    generate_structured, provider_label,
)
from .progress import ActivityIndicator

LOGGER = logging.getLogger("anki_pipeline.semantic")
SEMANTIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["segments"],
    "properties": {"segments": {
        "type": "array", "minItems": 1,
        "items": {
            "type": "object",
            "required": ["title", "summary", "transcript_segment_ids"],
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "transcript_segment_ids": {
                    "type": "array", "minItems": 1,
                    "items": {"type": "integer"},
                },
            },
        },
    }},
}


class SemanticSegmentationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SegmentWindow:
    context: list[TranscriptSegment]
    core_ids: list[int]


@dataclass(frozen=True)
class ProposedBlock:
    title: str
    summary: str
    transcript_segment_ids: list[int]


def load_transcript(path: Path) -> tuple[str, list[TranscriptSegment]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            TranscriptSegment(
                id=int(item["id"]),
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
            )
            for item in data["segments"]
        ]
        source = str(data["source"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SemanticSegmentationError(
            f"Не удалось прочитать транскрипцию {path}: {exc}"
        ) from exc
    if not segments:
        raise SemanticSegmentationError("Транскрипция не содержит сегментов.")
    ids = [segment.id for segment in segments]
    if ids != sorted(set(ids)):
        raise SemanticSegmentationError(
            "ID исходной транскрипции должны быть уникальными и упорядоченными."
        )
    return source, segments


def build_windows(
    segments: list[TranscriptSegment], *, max_chars: int = 7000, overlap: int = 3
) -> list[SegmentWindow]:
    windows: list[SegmentWindow] = []
    start = 0
    while start < len(segments):
        end = start
        used_chars = 0
        while end < len(segments):
            segment_size = len(segments[end].text) + 32
            if end > start and used_chars + segment_size > max_chars:
                break
            used_chars += segment_size
            end += 1
        context_start = max(0, start - overlap)
        context_end = min(len(segments), end + overlap)
        windows.append(
            SegmentWindow(
                context=segments[context_start:context_end],
                core_ids=[segment.id for segment in segments[start:end]],
            )
        )
        start = end
    return windows


def build_prompt(
    window: SegmentWindow, format_error: str | None = None,
    target_evidence: dict[int, list[str]] | None = None,
    output_language: str = "source",
) -> str:
    core_ids = set(window.core_ids)
    lines = [
        f"{segment.id} [{'CORE' if segment.id in core_ids else 'CONTEXT'}]: {segment.text}"
        + (
            "\n  TARGET_LANGUAGE: " + " | ".join((target_evidence or {}).get(segment.id, []))
            if (target_evidence or {}).get(segment.id) else ""
        )
        for segment in window.context
    ]
    correction = (
        f"\nПредыдущий ответ был отклонён: {format_error}\nИсправь формат и покрытие ID."
        if format_error
        else ""
    )
    language_instruction = {
        "en": "Write title and summary in English.",
        "ru": "Write title and summary in Russian.",
        "source": "Write title and summary in the primary transcript's language.",
    }.get(output_language)
    if language_instruction is None:
        raise SemanticSegmentationError("output language must be source, ru or en.")
    return f"""Раздели CORE-сегменты транскрипции на последовательные смысловые блоки.
CONTEXT дан только для понимания границ; не включай его ID в результат.
Верни только JSON-объект вида:
{{"segments":[{{"title":"краткая тема","summary":"1-2 предложения","transcript_segment_ids":[1,2]}}]}}
Правила:
- используй каждый CORE ID ровно один раз, строго по порядку;
- блоки непустые и состоят только из соседних CORE ID;
- title и summary должны быть непустыми;
- не возвращай исходный текст, start или end;
- {language_instruction}{correction}

CORE IDs: {window.core_ids}
Транскрипция:
{chr(10).join(lines)}"""


def load_target_evidence(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result: dict[int, list[str]] = {}
        for segment in data["segments"]:
            text = str(segment["text"]).strip()
            if not text:
                continue
            for source_id in segment["source_transcript_segment_ids"]:
                result.setdefault(int(source_id), []).append(text)
        return result
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SemanticSegmentationError(
            f"Не удалось прочитать target_transcript.json: {exc}"
        ) from exc


def parse_proposals(data: dict[str, Any], expected_ids: list[int]) -> list[ProposedBlock]:
    raw_blocks = data.get("segments")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise SemanticSegmentationError("Поле segments должно быть непустым массивом.")
    proposals: list[ProposedBlock] = []
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            raise SemanticSegmentationError("Каждый смысловой блок должен быть объектом.")
        title = raw.get("title")
        summary = raw.get("summary")
        ids = raw.get("transcript_segment_ids")
        if not isinstance(title, str) or not title.strip():
            raise SemanticSegmentationError("Название смыслового блока пустое.")
        if not isinstance(summary, str) or not summary.strip():
            raise SemanticSegmentationError("Summary смыслового блока пустое.")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, int) or isinstance(item, bool) for item in ids)
        ):
            raise SemanticSegmentationError("Блок содержит некорректный список ID.")
        proposals.append(ProposedBlock(title.strip(), summary.strip(), ids))

    proposals = trim_context_ids(proposals, expected_ids)
    proposals = trim_repeated_boundary_ids(proposals)

    actual_ids = [item for proposal in proposals for item in proposal.transcript_segment_ids]
    if actual_ids != expected_ids:
        raise SemanticSegmentationError(
            f"Нарушено покрытие ID: ожидалось {expected_ids}, получено {actual_ids}."
        )
    return proposals


def trim_repeated_boundary_ids(
    proposals: list[ProposedBlock],
) -> list[ProposedBlock]:
    """Remove an ID repeated only across the boundary of adjacent blocks."""
    normalized: list[ProposedBlock] = []
    for proposal in proposals:
        ids = proposal.transcript_segment_ids
        if (
            normalized
            and normalized[-1].transcript_segment_ids[-1] == ids[0]
        ):
            ids = ids[1:]
        if ids:
            normalized.append(
                ProposedBlock(proposal.title, proposal.summary, ids)
            )
    return normalized


def trim_context_ids(
    proposals: list[ProposedBlock], expected_ids: list[int]
) -> list[ProposedBlock]:
    """Remove context IDs that the model attached to the result's outer edges."""
    if not expected_ids:
        return proposals

    expected = set(expected_ids)
    trimmed: list[ProposedBlock] = []
    for proposal in proposals:
        ids = proposal.transcript_segment_ids
        start = 0
        end = len(ids)
        while start < end and ids[start] not in expected:
            start += 1
        while end > start and ids[end - 1] not in expected:
            end -= 1
        core_ids = ids[start:end]
        if any(item not in expected for item in core_ids):
            continue
        if core_ids:
            trimmed.append(
                ProposedBlock(proposal.title, proposal.summary, core_ids)
            )
    return trimmed


def merge_adjacent_blocks(blocks: list[ProposedBlock]) -> list[ProposedBlock]:
    merged: list[ProposedBlock] = []
    for block in blocks:
        if merged and merged[-1].title.casefold() == block.title.casefold():
            previous = merged[-1]
            merged[-1] = ProposedBlock(
                title=previous.title,
                summary=f"{previous.summary} {block.summary}".strip(),
                transcript_segment_ids=(
                    previous.transcript_segment_ids + block.transcript_segment_ids
                ),
            )
        else:
            merged.append(block)
    return merged


def assemble_result(
    source: str,
    model: str,
    transcript_segments: list[TranscriptSegment],
    proposals: list[ProposedBlock],
    target_evidence: dict[int, list[str]] | None = None,
    output_language: str = "source",
) -> SemanticTranscript:
    by_id = {segment.id: segment for segment in transcript_segments}
    expected_ids = [segment.id for segment in transcript_segments]
    actual_ids = [item for proposal in proposals for item in proposal.transcript_segment_ids]
    if actual_ids != expected_ids:
        raise SemanticSegmentationError(
            f"Итоговые ID не покрывают транскрипцию ровно один раз: {actual_ids}."
        )

    result_segments: list[SemanticSegment] = []
    for index, proposal in enumerate(proposals):
        originals = [by_id[item] for item in proposal.transcript_segment_ids]
        result_segments.append(
            SemanticSegment(
                id=index,
                title=proposal.title,
                start=originals[0].start,
                end=originals[-1].end,
                summary=proposal.summary,
                transcript_segment_ids=proposal.transcript_segment_ids,
                text=" ".join(segment.text.strip() for segment in originals).strip(),
                target_language_text=" ".join(
                    text
                    for source_id in proposal.transcript_segment_ids
                    for text in (target_evidence or {}).get(source_id, [])
                ).strip(),
            )
        )
    return SemanticTranscript(
        source=source, model=model, segments=result_segments,
        output_language=output_language,
    )


def save_json_atomic(data: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def save_diagnostic(path: Path, error_message: str, responses: list[str]) -> None:
    save_json_atomic(
        {"error": error_message, "ollama_responses": responses},
        path,
    )


def semantic_segment_transcript(
    transcript_path: Path,
    destination: Path,
    diagnostic_path: Path,
    client: StructuredLLM,
    *,
    target_transcript_path: Path | None = None,
    output_language: str = "source",
) -> SemanticTranscript:
    source, transcript_segments = load_transcript(transcript_path)
    all_proposals: list[ProposedBlock] = []
    windows = build_windows(transcript_segments)
    target_evidence = load_target_evidence(target_transcript_path)

    for window_number, window in enumerate(windows, start=1):
        LOGGER.info("Смысловая сегментация: окно %d/%d", window_number, len(windows))
        responses: list[str] = []
        format_error: str | None = None
        for attempt in range(2):
            try:
                retry_label = " (повтор)" if attempt else ""
                with ActivityIndicator(
                    f"{provider_label(client)}: окно {window_number}/{len(windows)}{retry_label}"
                ):
                    data, raw = generate_structured(
                        client, build_prompt(
                            window, format_error, target_evidence, output_language
                        ),
                        schema=SEMANTIC_RESPONSE_SCHEMA,
                    )
                responses.append(raw)
                proposals = parse_proposals(data, window.core_ids)
                all_proposals.extend(proposals)
                break
            except LLMInvalidResponseError as exc:
                responses.append(exc.raw_response)
                format_error = str(exc)
            except SemanticSegmentationError as exc:
                format_error = str(exc)
            except LLMError:
                raise
            if attempt == 1:
                message = f"Окно {window_number}: {format_error}"
                save_diagnostic(diagnostic_path, message, responses)
                raise SemanticSegmentationError(
                    f"{provider_label(client)} дважды вернул некорректный результат. {message} "
                    f"Диагностика: {diagnostic_path}"
                )

    result = assemble_result(
        source,
        client.model,
        transcript_segments,
        merge_adjacent_blocks(all_proposals),
        target_evidence,
        output_language,
    )
    save_json_atomic(result.to_dict(), destination)
    diagnostic_path.unlink(missing_ok=True)
    return result
