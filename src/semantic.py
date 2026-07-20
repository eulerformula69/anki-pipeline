from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SemanticSegment, SemanticTranscript, TranscriptSegment
from .llm import LLMError, LLMInvalidResponseError, StructuredLLM, generate_structured
from .progress import ActivityIndicator

LOGGER = logging.getLogger("anki_pipeline.semantic")


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


def build_prompt(window: SegmentWindow, format_error: str | None = None) -> str:
    core_ids = set(window.core_ids)
    lines = [
        f"{segment.id} [{'CORE' if segment.id in core_ids else 'CONTEXT'}]: {segment.text}"
        for segment in window.context
    ]
    correction = (
        f"\nПредыдущий ответ был отклонён: {format_error}\nИсправь формат и покрытие ID."
        if format_error
        else ""
    )
    return f"""Раздели CORE-сегменты транскрипции на последовательные смысловые блоки.
CONTEXT дан только для понимания границ; не включай его ID в результат.
Верни только JSON-объект вида:
{{"segments":[{{"title":"краткая тема","summary":"1-2 предложения","transcript_segment_ids":[1,2]}}]}}
Правила:
- используй каждый CORE ID ровно один раз, строго по порядку;
- блоки непустые и состоят только из соседних CORE ID;
- title и summary должны быть непустыми;
- не возвращай исходный текст, start или end;
- пиши title и summary на русском языке.{correction}

CORE IDs: {window.core_ids}
Транскрипция:
{chr(10).join(lines)}"""


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

    actual_ids = [item for proposal in proposals for item in proposal.transcript_segment_ids]
    if actual_ids != expected_ids:
        raise SemanticSegmentationError(
            f"Нарушено покрытие ID: ожидалось {expected_ids}, получено {actual_ids}."
        )
    return proposals


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
            )
        )
    return SemanticTranscript(source=source, model=model, segments=result_segments)


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
) -> SemanticTranscript:
    source, transcript_segments = load_transcript(transcript_path)
    all_proposals: list[ProposedBlock] = []
    windows = build_windows(transcript_segments)

    for window_number, window in enumerate(windows, start=1):
        LOGGER.info("Смысловая сегментация: окно %d/%d", window_number, len(windows))
        responses: list[str] = []
        format_error: str | None = None
        for attempt in range(2):
            try:
                retry_label = " (повтор)" if attempt else ""
                with ActivityIndicator(
                    f"Ollama: окно {window_number}/{len(windows)}{retry_label}"
                ):
                    data, raw = generate_structured(
                        client, build_prompt(window, format_error), schema={"type": "object"}
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
                    f"Ollama дважды вернул некорректный результат. {message} "
                    f"Диагностика: {diagnostic_path}"
                )

    result = assemble_result(
        source,
        client.model,
        transcript_segments,
        merge_adjacent_blocks(all_proposals),
    )
    save_json_atomic(result.to_dict(), destination)
    diagnostic_path.unlink(missing_ok=True)
    return result
