from pathlib import Path

from src.paths import build_result_paths, source_identity


def test_build_result_paths() -> None:
    paths = build_result_paths(Path("input/lesson.mp4"), Path("results"))

    assert paths.directory == Path("results/lesson")
    assert paths.audio == Path("results/lesson/audio.wav")
    assert paths.transcript_json == Path("results/lesson/transcript.json")
    assert paths.transcript_txt == Path("results/lesson/transcript.txt")
    assert paths.knowledge_items_json == Path("results/lesson/knowledge_items.json")
    assert paths.knowledge_diagnostic_json == Path(
        "results/lesson/knowledge_items_diagnostic.json"
    )
    assert paths.knowledge_review_json == Path("results/lesson/knowledge_review.json")
    assert paths.card_candidates_json == Path("results/lesson/card_candidates.json")
    assert paths.target_transcript_json == Path("results/lesson/target_transcript.json")
    assert paths.target_audio_directory == Path("results/lesson/target_audio")
    assert paths.card_media_json == Path("results/lesson/card_media.json")
    assert paths.media_directory == Path("results/lesson/media")
    assert paths.anki_deck == Path("results/lesson/lesson.apkg")
    assert paths.anki_export_manifest_json == Path(
        "results/lesson/anki_export_manifest.json"
    )


def test_youtube_manifest_gives_unique_project_and_source_identity(tmp_path) -> None:
    video = tmp_path / "input" / "downloads" / "Youtube" / "abc123" / "video.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    (video.parent / "source_manifest.json").write_text(
        '{"source_id":"abc123","kind":"youtube"}', encoding="utf-8"
    )

    paths = build_result_paths(video, tmp_path / "output")

    assert paths.directory == tmp_path / "output" / "abc123"
    assert paths.anki_deck.name == "abc123.apkg"
    assert source_identity(video) == "youtube:abc123"
