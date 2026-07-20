from pathlib import Path
import subprocess
import sys

import pytest

from src.pipeline import run_pipeline, validate_input


def test_missing_input_file_raises_clear_error(tmp_path) -> None:
    missing = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError, match="Входной файл не найден"):
        validate_input(missing)


def test_existing_input_file_is_valid(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    validate_input(video)


def test_pipeline_supports_direct_script_invocation() -> None:
    result = subprocess.run(
        [sys.executable, "src/pipeline.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--device" in result.stdout


def test_extract_knowledge_does_not_run_expensive_stages(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    video = Path("input/video.webm")
    video.parent.mkdir()
    video.touch()
    result_dir = Path("output/video")
    result_dir.mkdir(parents=True)
    (result_dir / "transcript.json").write_text("{}", encoding="utf-8")
    (result_dir / "semantic_segments.json").write_text("{}", encoding="utf-8")
    called = False

    def fake_extract(*args, **kwargs) -> None:
        nonlocal called
        called = True

    def forbidden(*args, **kwargs) -> None:
        raise AssertionError("Дорогой этап не должен запускаться")

    monkeypatch.setattr("src.pipeline.extract_knowledge", fake_extract)
    monkeypatch.setattr("src.pipeline.ensure_ffmpeg", forbidden)
    monkeypatch.setattr("src.pipeline.transcribe_audio", forbidden)

    result = run_pipeline(video, extract_knowledge_items=True)

    assert called
    assert result == result_dir / "knowledge_items.json"


def test_existing_knowledge_is_reused(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    video = Path("input/video.webm")
    video.parent.mkdir()
    video.touch()
    destination = Path("output/video/knowledge_items.json")
    destination.parent.mkdir(parents=True)
    destination.write_text('{"valid": true}', encoding="utf-8")

    monkeypatch.setattr(
        "src.pipeline.extract_knowledge",
        lambda *args, **kwargs: pytest.fail("Ollama не должен запускаться"),
    )

    assert run_pipeline(video, extract_knowledge_items=True) == destination


def test_extract_knowledge_does_not_require_original_video(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    video = Path("input/deleted-video.webm")
    result_dir = Path("output/deleted-video")
    result_dir.mkdir(parents=True)
    (result_dir / "transcript.json").write_text("{}", encoding="utf-8")
    (result_dir / "semantic_segments.json").write_text("{}", encoding="utf-8")
    called = False

    def fake_extract(*args, **kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("src.pipeline.extract_knowledge", fake_extract)

    run_pipeline(video, extract_knowledge_items=True)

    assert called


def test_review_does_not_run_previous_stages_or_require_video(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result_dir = Path("output/video")
    result_dir.mkdir(parents=True)
    (result_dir / "knowledge_items.json").write_text("{}", encoding="utf-8")
    called = False

    def fake_review(*args, **kwargs) -> None:
        nonlocal called
        called = True

    def forbidden(*args, **kwargs) -> None:
        raise AssertionError("Предыдущий этап не должен запускаться")

    monkeypatch.setattr("src.pipeline.review_knowledge", fake_review)
    monkeypatch.setattr("src.pipeline.ensure_ffmpeg", forbidden)
    monkeypatch.setattr("src.pipeline.transcribe_audio", forbidden)

    result = run_pipeline(Path("input/video.mkv"), review_knowledge_items=True)

    assert called
    assert result == result_dir / "knowledge_review.json"


def test_cards_do_not_run_previous_stages_or_require_video(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result_dir = Path("output/video")
    result_dir.mkdir(parents=True)
    (result_dir / "knowledge_items.json").write_text("{}", encoding="utf-8")
    (result_dir / "knowledge_review.json").write_text("{}", encoding="utf-8")
    called = False

    def fake_cards(*args, **kwargs) -> None:
        nonlocal called
        called = True

    def forbidden(*args, **kwargs) -> None:
        raise AssertionError("Предыдущий этап не должен запускаться")

    monkeypatch.setattr("src.pipeline.generate_card_candidates", fake_cards)
    monkeypatch.setattr("src.pipeline.ensure_ffmpeg", forbidden)
    monkeypatch.setattr("src.pipeline.transcribe_audio", forbidden)

    result = run_pipeline(Path("input/video.mkv"), generate_cards=True)

    assert called
    assert result == result_dir / "card_candidates.json"


def test_card_images_are_independent_and_reused(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result_dir = Path("output/video")
    result_dir.mkdir(parents=True)
    (result_dir / "transcript.json").write_text("{}", encoding="utf-8")
    (result_dir / "card_candidates.json").write_text("{}", encoding="utf-8")
    called = []
    monkeypatch.setattr("src.pipeline.extract_card_images", lambda *a, **k: called.append((a, k)))
    forbidden = lambda *a, **k: pytest.fail("Предыдущий этап не должен запускаться")
    monkeypatch.setattr("src.pipeline.ensure_ffmpeg", forbidden)
    monkeypatch.setattr("src.pipeline.transcribe_audio", forbidden)
    video = Path("input/video.mkv")
    assert run_pipeline(video, extract_card_images_stage=True, frames_per_card=3, image_policy="never") == result_dir / "card_media.json"
    assert called[0][1] == {"frames_per_card": 3, "image_policy": "never"}
    (result_dir / "card_media.json").write_text("{}", encoding="utf-8")
    called.clear()
    assert run_pipeline(video, extract_card_images_stage=True) == result_dir / "card_media.json"
    assert not called


def test_anki_export_is_independent_and_reused(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result_dir = Path("output/video")
    result_dir.mkdir(parents=True)
    (result_dir / "card_candidates.json").write_text("{}", encoding="utf-8")
    called = []
    monkeypatch.setattr("src.pipeline.build_anki_deck", lambda *a, **k: called.append((a, k)))
    forbidden = lambda *a, **k: pytest.fail("Предыдущий этап не должен запускаться")
    monkeypatch.setattr("src.pipeline.ensure_ffmpeg", forbidden)
    monkeypatch.setattr("src.pipeline.transcribe_audio", forbidden)
    destination = result_dir / "video.apkg"
    assert run_pipeline(
        Path("input/video.mkv"), build_anki_deck_stage=True, deck_name="Course"
    ) == destination
    assert called[0][1] == {"deck_name": "Course"}
    destination.write_bytes(b"apkg")
    (result_dir / "anki_export_manifest.json").write_text("{}", encoding="utf-8")
    called.clear()
    assert run_pipeline(Path("input/video.mkv"), build_anki_deck_stage=True) == destination
    assert not called
