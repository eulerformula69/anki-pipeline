from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .audio import ensure_ffmpeg
from .models import TargetTranscript, TargetTranscriptSegment, TranscriptSegment
from .semantic import save_json_atomic
from .transcription import resolve_device
from .target_expressions import romanization_matches_primary

LOGGER = logging.getLogger("anki_pipeline.target_transcription")


class TargetTranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetAudioRange:
    start: float
    end: float
    source_transcript_segment_ids: list[int]


def load_candidate_source_ids(
    expressions_path: Path, primary_text: dict[int, str] | None = None,
) -> set[int]:
    try:
        data = json.loads(expressions_path.read_text(encoding="utf-8"))
        expressions = data["expressions"]
        has_types = any("type" in expression for expression in expressions)
        selected = [
            expression for expression in expressions
            if not has_types
            or expression.get("type") == "example_sentence"
            or expression.get("verification") in {"verified", "transcribed"}
        ]
        result: set[int] = set()
        for expression in selected:
            romanization = str(expression.get("romanization", ""))
            for segment_id in expression["source_transcript_segment_ids"]:
                source_id = int(segment_id)
                if primary_text is None or romanization_matches_primary(
                    romanization, primary_text.get(source_id, "")
                ):
                    result.add(source_id)
        return result
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TargetTranscriptionError(
            f"Не удалось прочитать target_expressions.json: {exc}"
        ) from exc


def load_primary_transcript(path: Path) -> tuple[str, float, list[TranscriptSegment]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            str(data["source"]),
            float(data["duration_seconds"]),
            [TranscriptSegment(**item) for item in data["segments"]],
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TargetTranscriptionError(f"Не удалось прочитать transcript.json: {exc}") from exc


def build_target_ranges(
    segments: list[TranscriptSegment], candidate_ids: set[int], *,
    duration: float, padding: float = 0.35, max_range_seconds: float = 30.0,
) -> list[TargetAudioRange]:
    selected = [segment for segment in segments if segment.id in candidate_ids]
    if not selected:
        raise TargetTranscriptionError(
            "В target_expressions.json нет таймкодов для второго прохода."
        )
    ranges: list[TargetAudioRange] = []
    group: list[TranscriptSegment] = []
    for segment in selected:
        can_join = (
            group
            and segment.id == group[-1].id + 1
            and segment.start - group[-1].end <= 1.5
            and segment.end - group[0].start <= max_range_seconds
        )
        if group and not can_join:
            ranges.append(_range_from_group(group, duration, padding))
            group = []
        group.append(segment)
    if group:
        ranges.append(_range_from_group(group, duration, padding))
    return ranges


def _range_from_group(
    group: list[TranscriptSegment], duration: float, padding: float
) -> TargetAudioRange:
    return TargetAudioRange(
        start=round(max(0.0, group[0].start - padding), 3),
        end=round(min(duration, group[-1].end + padding), 3),
        source_transcript_segment_ids=[segment.id for segment in group],
    )


def extract_target_clip(
    audio: Path, destination: Path, audio_range: TargetAudioRange,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ensure_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{audio_range.start:.3f}", "-t",
        f"{audio_range.end - audio_range.start:.3f}", "-i", str(audio),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or "неизвестная ошибка FFmpeg"
        raise TargetTranscriptionError(
            f"Не удалось вырезать фрагмент {audio_range.start:.1f}–{audio_range.end:.1f}: {details}"
        ) from exc


def transcribe_target_ranges(
    audio: Path, transcript_path: Path, expressions_path: Path,
    destination: Path, clips_directory: Path, *, language: str,
    model_name: str = "small", device: str = "auto", padding: float = 0.35,
) -> TargetTranscript:
    try:
        from faster_whisper import WhisperModel
        from tqdm import tqdm
    except ImportError as exc:
        raise TargetTranscriptionError(
            "Для второго прохода нужны faster-whisper и tqdm. Установите requirements.txt."
        ) from exc
    source, duration, primary_segments = load_primary_transcript(transcript_path)
    ranges = build_target_ranges(
        primary_segments, load_candidate_source_ids(
            expressions_path, {segment.id: segment.text for segment in primary_segments}
        ),
        duration=duration, padding=padding,
    )
    selected_device, compute_type = resolve_device(device)
    try:
        model = WhisperModel(model_name, device=selected_device, compute_type=compute_type)
    except Exception as exc:
        raise TargetTranscriptionError(f"Не удалось загрузить Whisper {model_name}: {exc}") from exc
    result_segments: list[TargetTranscriptSegment] = []
    with tqdm(total=len(ranges), desc=f"Транскрибация {language}", unit="фрагм.") as progress:
        for index, audio_range in enumerate(ranges):
            clip = clips_directory / f"clip_{index + 1:04d}.wav"
            extract_target_clip(audio, clip, audio_range)
            try:
                raw_segments, info = model.transcribe(
                    str(clip), language=language, beam_size=5,
                    vad_filter=True, condition_on_previous_text=False,
                )
                text = " ".join(item.text.strip() for item in raw_segments if item.text.strip())
            except Exception as exc:
                raise TargetTranscriptionError(
                    f"Ошибка второго прохода на фрагменте {index + 1}: {exc}"
                ) from exc
            if text:
                result_segments.append(TargetTranscriptSegment(
                    id=len(result_segments), start=audio_range.start, end=audio_range.end,
                    text=text,
                    source_transcript_segment_ids=audio_range.source_transcript_segment_ids,
                    language_probability=round(float(info.language_probability), 4),
                ))
            progress.update(1)
    result = TargetTranscript(
        source=source, language=language, duration_seconds=duration,
        model=model_name, segments=result_segments,
    )
    save_json_atomic(result.to_dict(), destination)
    return result
