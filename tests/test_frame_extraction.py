import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from src.frame_extraction import (
    FrameCandidate,
    FrameExtractionError,
    calculate_frame_metrics,
    calculate_frame_similarity,
    calculate_quality_score,
    choose_frame_timestamps,
    compress_existing_card_images,
    extract_card_images,
    extract_frame,
    load_media_sources,
    resolve_card_time_range,
    select_best_frame,
)
from src.models import FrameMetrics, TranscriptSegment


def test_existing_card_images_are_compressed_to_ten_kibibytes(tmp_path) -> None:
    import numpy as np

    media = tmp_path / "media" / "c_0001.jpg"
    media.parent.mkdir()
    pixels = np.random.default_rng(42).integers(0, 256, (900, 1600, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(media)
    manifest = tmp_path / "card_media.json"
    manifest.write_text(json.dumps({"cards": [{
        "card_id": "c_0001", "image": {
            "path": "media/c_0001.jpg", "width": 1600, "height": 900,
        },
    }]}), encoding="utf-8")

    assert compress_existing_card_images(manifest) == 1
    assert media.stat().st_size <= 10 * 1024
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    with Image.open(media) as compressed:
        dimensions = [saved["cards"][0]["image"][key] for key in ("width", "height")]
        assert dimensions == list(compressed.size)


def write_sources(root: Path) -> tuple[Path, Path]:
    transcript = root / "transcript.json"
    cards = root / "card_candidates.json"
    transcript.write_text(json.dumps({"source": "video.mkv", "segments": [
        {"id": 1, "start": 2.0, "end": 3.0, "text": "a"},
        {"id": 2, "start": 5.0, "end": 8.0, "text": "b"},
    ]}), encoding="utf-8")
    cards.write_text(json.dumps({"card_candidates": [
        {"card_id": "c_0001", "source_transcript_segment_ids": [2, 1]}
    ]}), encoding="utf-8")
    return transcript, cards


def test_sources_and_time_range_use_original_segments(tmp_path) -> None:
    transcript, cards = write_sources(tmp_path)
    _, segments, loaded_cards = load_media_sources(transcript, cards)
    assert resolve_card_time_range(
        loaded_cards[0].source_transcript_segment_ids, {item.id: item for item in segments}
    ) == (2.0, 8.0)


def test_missing_and_empty_source_ids_are_rejected(tmp_path) -> None:
    by_id = {1: TranscriptSegment(1, 0, 1, "a")}
    with pytest.raises(FrameExtractionError, match="пустым"):
        resolve_card_time_range([], by_id)
    with pytest.raises(FrameExtractionError, match="Неизвестный"):
        resolve_card_time_range([2], by_id)


def test_timestamps_are_centered_uniform_and_deterministic() -> None:
    assert choose_frame_timestamps(10, 20, 1) == [15]
    result = choose_frame_timestamps(10, 20, 5)
    assert result == [10.25, 12.625, 15.0, 17.375, 19.75]
    assert result == choose_frame_timestamps(10, 20, 5)
    assert all(10 <= value <= 20 for value in result)
    assert choose_frame_timestamps(3, 3, 2) == [3, 3]


def test_invalid_timestamp_ranges_are_rejected() -> None:
    with pytest.raises(FrameExtractionError):
        choose_frame_timestamps(-1, 2, 1)
    with pytest.raises(FrameExtractionError):
        choose_frame_timestamps(2, 1, 1)


def test_metrics_detect_black_sharpness_and_similarity(tmp_path) -> None:
    black = tmp_path / "black.png"
    checker = tmp_path / "checker.png"
    blurred = tmp_path / "blurred.png"
    Image.new("L", (64, 64), 0).save(black)
    Image.new("L", (64, 64), 255).save(tmp_path / "white.png")
    image = Image.new("L", (64, 64))
    image.putdata([255 if (x // 4 + y // 4) % 2 else 0 for y in range(64) for x in range(64)])
    image.save(checker)
    image.filter(ImageFilter.GaussianBlur(4)).save(blurred)
    assert calculate_frame_metrics(black).dark_ratio == 1.0
    assert calculate_frame_metrics(checker).sharpness > calculate_frame_metrics(blurred).sharpness
    assert calculate_frame_similarity(checker, checker) == 1.0
    assert calculate_frame_similarity(black, tmp_path / "white.png") < 0.1
    assert 0 <= calculate_quality_score(calculate_frame_metrics(checker)) <= 1


def candidate(path: Path, score: float, timestamp: float, sharpness: float = 100) -> FrameCandidate:
    return FrameCandidate(path, timestamp, FrameMetrics(120, 0, sharpness), 64, 64, score, abs(timestamp - 5))


def test_selection_policies_and_tie_break(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("L", (32, 32), 100).save(first)
    Image.new("L", (32, 32), 150).save(second)
    candidates = [candidate(first, .8, 2), candidate(second, .8, 4)]
    assert select_best_frame(candidates, policy="never") is None
    assert select_best_frame(candidates, policy="always").timestamp == 4
    assert select_best_frame([candidate(first, .1, 2)], policy="auto") is None
    black = FrameCandidate(first, 2, FrameMetrics(0, 1, 100), 32, 32, .9, 3)
    assert select_best_frame([black], policy="always") is None


def test_extract_frame_uses_argument_list_and_wraps_ffmpeg_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.frame_extraction.shutil.which", lambda _: "ffmpeg")
    seen = {}
    def fake_run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        raise subprocess.CalledProcessError(1, command, stderr="broken input")
    monkeypatch.setattr("src.frame_extraction.subprocess.run", fake_run)
    with pytest.raises(FrameExtractionError, match="broken input"):
        extract_frame(Path("folder with space/video.mkv"), 1.25, tmp_path / "out.jpg")
    assert isinstance(seen["command"], list)
    assert str(Path("folder with space/video.mkv")) in seen["command"]
    assert "shell" not in seen
    assert seen["check"] and seen["capture_output"] and seen["text"]


def test_never_creates_manifest_without_ffmpeg(tmp_path, monkeypatch) -> None:
    transcript, cards = write_sources(tmp_path)
    monkeypatch.setattr("src.frame_extraction.extract_frame", lambda *a, **k: pytest.fail("FFmpeg called"))
    destination = tmp_path / "card_media.json"
    result = extract_card_images(Path("missing.mkv"), transcript, cards, destination, tmp_path / "media", image_policy="never")
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert result.cards[0].image is None
    assert data["cards"][0]["image"] is None
    assert data["cards"][0]["selection_method"] == "disabled_by_policy"
