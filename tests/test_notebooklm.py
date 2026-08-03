import json

import pytest

from src.notebooklm import NotebookLMImportError, import_notebooklm_cards


def write_transcript(path):
    path.write_text(json.dumps({"source": "lesson.mp4", "segments": [
        {"id": 1, "start": 0, "end": 4, "text": "The past tense of go is went."},
        {"id": 2, "start": 4, "end": 8, "text": "Use went when you describe a completed trip."},
    ]}), encoding="utf-8")


def test_import_preserves_notebooklm_text_and_links_the_best_transcript_segment(tmp_path):
    transcript = tmp_path / "transcript.json"
    destination = tmp_path / "cards.json"
    write_transcript(transcript)

    result = import_notebooklm_cards(
        "2351 | What is the past tense of go? | The past tense of go is went.\n",
        transcript, destination,
    )

    card = result.card_candidates[0]
    assert card.card_id == "n_2351"
    assert card.front == "What is the past tense of go?"
    assert card.back == "The past tense of go is went."
    assert card.source_transcript_segment_ids == [1]
    assert json.loads(destination.read_text(encoding="utf-8"))["model"] == "notebooklm-import-v1"


def test_import_rejects_malformed_lines(tmp_path):
    transcript = tmp_path / "transcript.json"
    write_transcript(transcript)

    with pytest.raises(NotebookLMImportError, match="три поля"):
        import_notebooklm_cards("2351 | only a question\n", transcript, tmp_path / "cards.json")
