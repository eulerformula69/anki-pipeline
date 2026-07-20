import json

import pytest
from fastapi.testclient import TestClient

from src.anki_export import AnkiExportError, build_anki_deck
from src.gui.app import create_app
from src.gui.jobs import JobManager
from src.gui.schemas import JobState


def test_export_rejects_draft_before_importing_genanki(tmp_path):
    candidates = {"source": "lesson.mp4", "card_candidates": [{
        "card_id": "c1", "knowledge_ids": ["k1"], "card_type": "basic",
        "front": "Q", "back": "A", "extra": "", "difficulty": 1,
        "importance": 1, "tags": [], "source_transcript_segment_ids": [1],
    }]}
    review = {"source": "lesson.mp4", "review_version": 1, "status": "draft", "approved_at": None, "cards": []}
    cards = tmp_path / "card_candidates.json"
    approved = tmp_path / "approved_cards.json"
    cards.write_text(json.dumps(candidates), encoding="utf-8")
    approved.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(AnkiExportError, match="одобрен"):
        build_anki_deck(cards, tmp_path / "expressions.json", tmp_path / "media.json", tmp_path / "deck.apkg", tmp_path / "manifest.json", approved_cards_path=approved)


def test_export_stays_on_review_and_passes_name_and_global_tags(tmp_path, monkeypatch):
    output = tmp_path / "output" / "video"
    output.mkdir(parents=True)
    candidates = {"source": "lesson.mp4", "card_candidates": [{
        "card_id": "c1", "knowledge_ids": ["k1"], "card_type": "basic",
        "front": "Q", "back": "A", "extra": "", "difficulty": 1,
        "importance": 1, "tags": [], "source_transcript_segment_ids": [1],
    }]}
    review = {
        "source": "lesson.mp4", "review_version": 1, "status": "approved",
        "updated_at": "now", "approved_at": "now", "cards": [{
            "card_id": "c1", "included": True, "front": "Q", "back": "A",
            "extra": "", "tags": [], "difficulty": 1, "importance": 1,
            "image": {"mode": "none"}, "overrides": {}, "reviewed": True,
            "changed": False, "updated_at": "now", "stale": False,
        }],
    }
    (output / "card_candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    (output / "approved_cards.json").write_text(json.dumps(review), encoding="utf-8")
    manager = JobManager(output_root=tmp_path / "output")
    manager.jobs["job1"] = JobState(
        "job1", "Lesson", str(tmp_path / "video.mkv"), str(output),
        status="awaiting_review", created_at="now",
    )
    captured = {}
    monkeypatch.setattr(
        "src.gui.routes.export_approved_deck",
        lambda video, name, tags: captured.update(name=name, tags=tags) or output / "deck.apkg",
    )
    client = TestClient(create_app(root=tmp_path, manager=manager))
    token = client.get("/").cookies["anki_csrf"]

    response = client.post(
        "/jobs/job1/export",
        data={"csrf": token, "deck_name": "Japanese Grammar", "global_tags": "japanese, lesson-24"},
        headers={"HX-Request": "true"}, follow_redirects=False,
    )

    assert response.status_code == 200
    assert "Скачать .apkg" in response.text
    assert "<html" not in response.text
    assert captured == {"name": "Japanese Grammar", "tags": ["japanese", "lesson-24"]}
