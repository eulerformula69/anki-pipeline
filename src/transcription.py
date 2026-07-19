from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import Transcript, TranscriptSegment


class TranscriptionError(RuntimeError):
    pass


def collect_segments(
    raw_segments: Iterable[object], duration: float
) -> list[TranscriptSegment]:
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise TranscriptionError(
            "tqdm не установлен. Выполните: pip install -r requirements.txt"
        ) from exc

    segments: list[TranscriptSegment] = []
    with tqdm(
        total=duration,
        desc="Транскрибация",
        unit="сек",
        dynamic_ncols=True,
    ) as progress:
        for index, segment in enumerate(raw_segments):
            segments.append(
                TranscriptSegment(
                    id=index,
                    start=round(segment.start, 3),
                    end=round(segment.end, 3),
                    text=segment.text.strip(),
                )
            )
            processed = min(float(segment.end), duration)
            progress.update(max(0.0, processed - progress.n))

        progress.update(max(0.0, duration - progress.n))

    return segments


def resolve_device(requested: str) -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "float16"

    try:
        import ctranslate2

        compute_types = ctranslate2.get_supported_compute_types("cuda")
        if "float16" in compute_types:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def transcribe_audio(
    audio: Path,
    source: Path,
    *,
    model_name: str,
    device: str,
    language: str | None,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper не установлен. Выполните: pip install -r requirements.txt"
        ) from exc

    selected_device, compute_type = resolve_device(device)
    try:
        model = WhisperModel(
            model_name, device=selected_device, compute_type=compute_type
        )
        raw_segments, info = model.transcribe(str(audio), language=language)
        segments = collect_segments(raw_segments, float(info.duration))
    except Exception as exc:
        hint = (
            " Попробуйте --device cpu, если CUDA настроена неправильно."
            if selected_device == "cuda"
            else ""
        )
        raise TranscriptionError(f"Ошибка транскрибации: {exc}.{hint}") from exc

    return Transcript(
        source=source.name,
        language=info.language,
        duration_seconds=round(float(info.duration), 3),
        model=model_name,
        segments=segments,
    )
