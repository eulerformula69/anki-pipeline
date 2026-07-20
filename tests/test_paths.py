from pathlib import Path

from src.paths import build_result_paths


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
