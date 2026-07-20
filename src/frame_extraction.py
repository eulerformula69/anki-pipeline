from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .models import (
    CardCandidate,
    CardImage,
    CardMedia,
    CardMediaCollection,
    FrameMetrics,
    TranscriptSegment,
)
from .semantic import save_json_atomic

GENERATOR = "ffmpeg-local-metrics-v1"
IMAGE_POLICIES = {"auto", "always", "never"}
MAX_FRAMES_PER_CARD = 20
DARK_RATIO_LIMIT = 0.95
MIN_SHARPNESS = 20.0
AUTO_QUALITY_THRESHOLD = 0.28
DUPLICATE_SIMILARITY = 0.985
SAFE_CARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


class FrameExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameCandidate:
    path: Path
    timestamp: float
    metrics: FrameMetrics
    width: int
    height: int
    quality_score: float
    distance_from_center: float


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FrameExtractionError(f"Не найден {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrameExtractionError(f"Не удалось прочитать {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise FrameExtractionError(f"{label} должен содержать JSON-объект.")
    return data


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrameExtractionError(f"{label} должен быть числом.")
    return float(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrameExtractionError(f"{label} должен быть целым числом.")
    return value


def load_media_sources(
    transcript_path: Path, cards_path: Path
) -> tuple[str, list[TranscriptSegment], list[CardCandidate]]:
    transcript_data = _read_json(transcript_path, "transcript.json")
    cards_data = _read_json(cards_path, "card_candidates.json")
    try:
        source = transcript_data["source"]
        raw_segments = transcript_data["segments"]
        raw_cards = cards_data["card_candidates"]
    except KeyError as exc:
        raise FrameExtractionError(f"Отсутствует обязательное поле {exc}.") from exc
    if not isinstance(source, str) or not isinstance(raw_segments, list) or not isinstance(raw_cards, list):
        raise FrameExtractionError("Некорректная структура входных JSON.")

    segments: list[TranscriptSegment] = []
    segment_ids: set[int] = set()
    for index, item in enumerate(raw_segments):
        try:
            segment_id = _integer(item["id"], f"segments[{index}].id")
            start = _number(item["start"], f"segments[{index}].start")
            end = _number(item["end"], f"segments[{index}].end")
            text = item["text"]
        except (KeyError, TypeError) as exc:
            raise FrameExtractionError(f"Некорректный сегмент {index}: {exc}") from exc
        if segment_id in segment_ids:
            raise FrameExtractionError(f"Повторяющийся transcript segment ID: {segment_id}.")
        if start < 0 or end < start or not isinstance(text, str):
            raise FrameExtractionError(f"Некорректные start/end/text сегмента {segment_id}.")
        segment_ids.add(segment_id)
        segments.append(TranscriptSegment(segment_id, start, end, text))

    cards: list[CardCandidate] = []
    card_ids: set[str] = set()
    for index, item in enumerate(raw_cards):
        try:
            card_id = item["card_id"]
            source_ids = item["source_transcript_segment_ids"]
        except (KeyError, TypeError) as exc:
            raise FrameExtractionError(f"Некорректная карточка {index}: {exc}") from exc
        if not isinstance(card_id, str) or not SAFE_CARD_ID.fullmatch(card_id):
            raise FrameExtractionError(f"Небезопасный card_id: {card_id!r}.")
        if card_id in card_ids:
            raise FrameExtractionError(f"Повторяющийся card_id: {card_id}.")
        if not isinstance(source_ids, list) or not source_ids:
            raise FrameExtractionError(f"Карточка {card_id} не содержит source IDs.")
        parsed_ids = [_integer(value, f"{card_id}.source ID") for value in source_ids]
        if len(parsed_ids) != len(set(parsed_ids)):
            raise FrameExtractionError(f"Карточка {card_id} содержит повторяющиеся source IDs.")
        unknown = [value for value in parsed_ids if value not in segment_ids]
        if unknown:
            raise FrameExtractionError(f"Карточка {card_id}: неизвестные transcript IDs {unknown}.")
        card_ids.add(card_id)
        cards.append(CardCandidate(
            card_id=card_id,
            knowledge_ids=[], card_type="", front="", back="", extra="",
            difficulty=1, importance=1, tags=[],
            source_transcript_segment_ids=parsed_ids,
        ))
    return source, segments, cards


def resolve_card_time_range(
    source_ids: list[int], transcript_by_id: dict[int, TranscriptSegment]
) -> tuple[float, float]:
    if not source_ids:
        raise FrameExtractionError("Список source transcript IDs не может быть пустым.")
    try:
        selected = [transcript_by_id[value] for value in source_ids]
    except KeyError as exc:
        raise FrameExtractionError(f"Неизвестный transcript segment ID: {exc.args[0]}.") from exc
    return min(item.start for item in selected), max(item.end for item in selected)


def choose_frame_timestamps(start: float, end: float, count: int) -> list[float]:
    if isinstance(count, bool) or not 1 <= count <= MAX_FRAMES_PER_CARD:
        raise FrameExtractionError(f"frames_per_card должен быть от 1 до {MAX_FRAMES_PER_CARD}.")
    if start < 0 or end < start:
        raise FrameExtractionError("Диапазон кадра должен удовлетворять 0 <= start <= end.")
    duration = end - start
    if duration <= 1e-9:
        return [start] * count
    padding = min(0.25, duration * 0.1)
    usable_start, usable_end = start + padding, end - padding
    if usable_end <= usable_start:
        return [(start + end) / 2.0] * count
    if count == 1:
        return [(usable_start + usable_end) / 2.0]
    step = (usable_end - usable_start) / (count - 1)
    return [usable_start + step * index for index in range(count)]


def extract_frame(video_path: Path, timestamp: float, destination: Path) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FrameExtractionError("FFmpeg не найден в PATH. Установите FFmpeg и повторите запуск.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.6f}",
        "-i", str(video_path), "-frames:v", "1", "-q:v", "2", "-y", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "неизвестная ошибка FFmpeg").strip()
        raise FrameExtractionError(f"FFmpeg не смог извлечь кадр {timestamp:.3f}: {details}") from exc


def calculate_frame_metrics(image_path: Path) -> FrameMetrics:
    try:
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise FrameExtractionError(f"Нечитаемый кадр {image_path}: {exc}") from exc
    if gray.size == 0:
        raise FrameExtractionError(f"Пустой кадр: {image_path}")
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    sharpness = float(np.var(dx) + np.var(dy))
    return FrameMetrics(
        brightness_mean=round(float(np.mean(gray)), 4),
        dark_ratio=round(float(np.mean(gray < 16)), 6),
        sharpness=round(sharpness, 4),
    )


def calculate_frame_similarity(first: Path, second: Path) -> float:
    def thumbnail(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            resized = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            return np.asarray(resized, dtype=np.float32)
    difference = float(np.mean(np.abs(thumbnail(first) - thumbnail(second))) / 255.0)
    return max(0.0, min(1.0, 1.0 - difference))


def calculate_quality_score(metrics: FrameMetrics) -> float:
    sharpness = min(metrics.sharpness / 500.0, 1.0)
    exposure = max(0.0, 1.0 - abs(metrics.brightness_mean - 127.5) / 127.5)
    darkness = max(0.0, 1.0 - metrics.dark_ratio)
    return round(max(0.0, min(1.0, 0.6 * sharpness + 0.25 * exposure + 0.15 * darkness)), 6)


def select_best_frame(candidates: list[FrameCandidate], *, policy: str) -> FrameCandidate | None:
    if policy not in IMAGE_POLICIES:
        raise FrameExtractionError(f"Неизвестная image_policy: {policy}.")
    if policy == "never":
        return None
    usable: list[FrameCandidate] = []
    for candidate in candidates:
        if candidate.metrics.dark_ratio >= DARK_RATIO_LIMIT:
            continue
        if policy == "auto" and candidate.metrics.sharpness < MIN_SHARPNESS:
            continue
        if policy == "auto" and any(
            calculate_frame_similarity(candidate.path, previous.path) >= DUPLICATE_SIMILARITY
            for previous in usable
        ):
            continue
        usable.append(candidate)
    if not usable:
        return None
    best = min(usable, key=lambda item: (-item.quality_score, item.distance_from_center, item.timestamp))
    if policy == "auto" and best.quality_score < AUTO_QUALITY_THRESHOLD:
        return None
    return best


def _old_managed_images(destination: Path) -> set[Path]:
    if not destination.exists():
        return set()
    try:
        data = json.loads(destination.read_text(encoding="utf-8"))
        return {
            destination.parent / item["image"]["path"]
            for item in data.get("cards", []) if isinstance(item.get("image"), dict)
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return set()


def extract_card_images(
    video_path: Path, transcript_path: Path, cards_path: Path, destination: Path,
    media_directory: Path, *, frames_per_card: int = 5, image_policy: str = "auto",
) -> CardMediaCollection:
    if image_policy not in IMAGE_POLICIES:
        raise FrameExtractionError(f"image_policy должна быть одной из {sorted(IMAGE_POLICIES)}.")
    if not 1 <= frames_per_card <= MAX_FRAMES_PER_CARD:
        raise FrameExtractionError(f"frames_per_card должен быть от 1 до {MAX_FRAMES_PER_CARD}.")
    source, segments, cards = load_media_sources(transcript_path, cards_path)
    by_id = {segment.id: segment for segment in segments}
    ranges = [(card, *resolve_card_time_range(card.source_transcript_segment_ids, by_id)) for card in cards]
    if image_policy != "never" and (not video_path.exists() or not video_path.is_file()):
        raise FrameExtractionError(f"Видеофайл не найден: {video_path}")
    prepared: list[CardMedia] = []
    staged: list[tuple[Path, Path]] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="card-media-", dir=destination.parent) as temporary:
        temp = Path(temporary)
        for card, start, end in ranges:
            if image_policy == "never":
                prepared.append(CardMedia(card.card_id, card.source_transcript_segment_ids, start, end, None, "disabled_by_policy", "image_policy_never"))
                continue
            candidates: list[FrameCandidate] = []
            center = (start + end) / 2.0
            for index, timestamp in enumerate(choose_frame_timestamps(start, end, frames_per_card)):
                path = temp / f"{card.card_id}-{index}.jpg"
                extract_frame(video_path, timestamp, path)
                metrics = calculate_frame_metrics(path)
                with Image.open(path) as image:
                    width, height = image.size
                candidates.append(FrameCandidate(path, timestamp, metrics, width, height, calculate_quality_score(metrics), abs(timestamp - center)))
            selected = select_best_frame(candidates, policy=image_policy)
            if selected is None:
                prepared.append(CardMedia(card.card_id, card.source_transcript_segment_ids, start, end, None, "no_suitable_frame", "all_candidates_rejected"))
                continue
            final = media_directory / f"{card.card_id}.jpg"
            staged.append((selected.path, final))
            image = CardImage(
                path=final.relative_to(destination.parent).as_posix(), timestamp=round(selected.timestamp, 6),
                width=selected.width, height=selected.height, selection_method="local_quality_score",
                quality_score=selected.quality_score, metrics=selected.metrics,
            )
            prepared.append(CardMedia(card.card_id, card.source_transcript_segment_ids, start, end, image))
        result = CardMediaCollection(source, GENERATOR, frames_per_card, image_policy, prepared)
        old_images = _old_managed_images(destination)
        media_directory.mkdir(parents=True, exist_ok=True)
        for source_path, final_path in staged:
            shutil.copy2(source_path, final_path)
        save_json_atomic(result.to_dict(), destination)
        current = {final for _, final in staged}
        for obsolete in old_images - current:
            if obsolete.parent == media_directory:
                obsolete.unlink(missing_ok=True)
        return result
