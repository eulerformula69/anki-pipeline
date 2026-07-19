from pathlib import Path

from src.paths import build_result_paths


def test_build_result_paths() -> None:
    paths = build_result_paths(Path("input/lesson.mp4"), Path("results"))

    assert paths.directory == Path("results/lesson")
    assert paths.audio == Path("results/lesson/audio.wav")
    assert paths.transcript_json == Path("results/lesson/transcript.json")
    assert paths.transcript_txt == Path("results/lesson/transcript.txt")

