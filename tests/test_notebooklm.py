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
        "1 | application | Complete: Yesterday I ___ home. | went\n",
        transcript, destination,
    )

    card = result.card_candidates[0]
    assert card.card_id == "n_0001"
    assert card.front == "Complete: Yesterday I ___ home."
    assert card.back == "went"
    assert card.card_type == "application"
    assert card.tags == ["LLM-source", "application"]
    assert card.source_transcript_segment_ids == [2]
    assert json.loads(destination.read_text(encoding="utf-8"))["model"] == "notebooklm-import-v1"


def test_import_can_append_cards_with_unique_ids(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        '{"source":"lesson.mp4","segments":[{"id":1,"start":0,"end":3,"text":"first second"}]}',
        encoding="utf-8",
    )
    destination = tmp_path / "cards.json"
    import_notebooklm_cards("1 | recall | First? | Yes", transcript, destination)

    result = import_notebooklm_cards(
        "1 | application | Second? | Also yes", transcript, destination, append=True
    )

    assert [card.card_id for card in result.card_candidates] == ["n_0001", "n_0002"]
    assert [card.front for card in result.card_candidates] == ["First?", "Second?"]


def test_import_without_video_uses_project_name_and_no_source_segments(tmp_path):
    destination = tmp_path / "card_candidates.json"

    result = import_notebooklm_cards(
        "1 | recall | What is retrieval practice? | Recalling information from memory.",
        None,
        destination,
        source="Learning science",
    )

    assert result.source == "Learning science"
    assert result.card_candidates[0].source_transcript_segment_ids == []


def test_import_preserves_inline_mathjax_verbatim(tmp_path):
    destination = tmp_path / "card_candidates.json"
    formula = r"\(L'(w)=2(w-4)+2\lambda w=4w-8\)"

    result = import_notebooklm_cards(
        f"1 | application | Find the optimum. | {formula}",
        None,
        destination,
        source="Regularization",
    )

    assert result.card_candidates[0].back == formula
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["card_candidates"][0]["back"] == formula


def test_import_rejects_malformed_lines(tmp_path):
    transcript = tmp_path / "transcript.json"
    write_transcript(transcript)

    with pytest.raises(NotebookLMImportError, match="id"):
        import_notebooklm_cards("only a question\n", transcript, tmp_path / "cards.json")


def test_import_preserves_learning_goals_and_assigns_its_own_ids(tmp_path):
    transcript = tmp_path / "transcript.json"
    write_transcript(transcript)

    result = import_notebooklm_cards(
        "source-a | recall | What is went? | The past tense of go.\n"
        "source-b | discrimination | Which form marks past tense: go or went? | went\n",
        transcript, tmp_path / "cards.json",
    )

    assert [card.card_id for card in result.card_candidates] == ["n_0001", "n_0002"]
    assert all(card.extra == "" for card in result.card_candidates)
    assert [card.card_type for card in result.card_candidates] == ["recall", "discrimination"]


def test_import_rejects_unknown_learning_goal(tmp_path):
    transcript = tmp_path / "transcript.json"
    write_transcript(transcript)

    with pytest.raises(NotebookLMImportError, match="неизвестный type"):
        import_notebooklm_cards(
            "1 | basic | Question? | Answer\n", transcript, tmp_path / "cards.json"
        )
