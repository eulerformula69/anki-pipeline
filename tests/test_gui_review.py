import json

import pytest
from fastapi.testclient import TestClient

from src.gui.app import create_app
from src.gui.jobs import JobManager
from src.gui.review_store import ReviewError, ReviewStore
from src.gui.schemas import JobState


def project(tmp_path):
    cards = {"source": "lesson.mp4", "card_candidates": [{
        "card_id": "c_1", "card_type": "basic", "front": "Q", "back": "A",
        "extra": "", "tags": ["lesson"], "difficulty": 2, "importance": 3,
        "knowledge_ids": ["k1"], "source_transcript_segment_ids": [1],
    }]}
    (tmp_path / "card_candidates.json").write_text(json.dumps(cards), encoding="utf-8")
    return tmp_path


def test_review_is_separate_and_edit_revokes_approval(tmp_path):
    output = project(tmp_path)
    store = ReviewStore()
    original = (output / "card_candidates.json").read_bytes()
    review = store.load_or_create(output)
    assert review["status"] == "draft"
    store.approve(output)
    store.save_card(output, "c_1", {"front": "Edited"})
    saved = json.loads((output / "approved_cards.json").read_text(encoding="utf-8"))
    assert saved["status"] == "draft" and saved["approved_at"] is None
    assert (output / "card_candidates.json").read_bytes() == original


def test_approval_rejects_empty_included_card(tmp_path):
    store = ReviewStore()
    output = project(tmp_path)
    store.load_or_create(output)
    store.save_card(output, "c_1", {"front": ""})
    with pytest.raises(ReviewError):
        store.approve(output)


def test_stale_card_with_removed_media_no_longer_blocks_review(tmp_path):
    output = project(tmp_path)
    media_dir = output / "media"
    media_dir.mkdir()
    image = media_dir / "c_1.jpg"
    image.write_bytes(b"image")
    (output / "card_media.json").write_text(json.dumps({
        "cards": [{"card_id": "c_1", "image": {"path": "media/c_1.jpg", "timestamp": 1.0}}]
    }), encoding="utf-8")
    store = ReviewStore()
    first = store.load_or_create(output)
    assert first["cards"][0]["image"]["mode"] == "automatic"

    image.unlink()
    replacement = {
        "source": "lesson.mp4", "card_candidates": [{
            "card_id": "c_2", "card_type": "basic", "front": "New", "back": "Card",
            "extra": "", "tags": [], "difficulty": 1, "importance": 1,
            "knowledge_ids": ["k2"], "source_transcript_segment_ids": [2],
        }],
    }
    (output / "card_candidates.json").write_text(json.dumps(replacement), encoding="utf-8")

    refreshed = store.load_or_create(output)
    stale = next(card for card in refreshed["cards"] if card["card_id"] == "c_1")
    assert stale["stale"] is True
    assert stale["included"] is False
    assert stale["image"] == {"mode": "none"}


def test_regeneration_refreshes_only_untouched_review_cards(tmp_path):
    output = project(tmp_path)
    store = ReviewStore()
    store.load_or_create(output)

    candidates = json.loads(
        (output / "card_candidates.json").read_text(encoding="utf-8")
    )
    candidates["card_candidates"][0]["front"] = "Regenerated question"
    (output / "card_candidates.json").write_text(
        json.dumps(candidates),
        encoding="utf-8",
    )

    refreshed = store.load_or_create(output)

    assert refreshed["cards"][0]["front"] == "Regenerated question"

    store.save_card(output, "c_1", {"front": "Manual question"})
    candidates["card_candidates"][0]["front"] = "Another generated question"
    (output / "card_candidates.json").write_text(
        json.dumps(candidates),
        encoding="utf-8",
    )

    preserved = store.load_or_create(output)

    assert preserved["cards"][0]["front"] == "Manual question"


def test_blank_card_is_added_and_cards_can_be_reordered(tmp_path):
    output = project(tmp_path)
    store = ReviewStore()
    store.load_or_create(output)

    card_id = store.add_blank_card(output)
    review = store.load_or_create(output)

    assert card_id == "manual_0001"
    assert [card["card_id"] for card in review["cards"]] == ["c_1", "manual_0001"]
    assert review["cards"][1]["front"] == ""

    store.move_card(output, card_id, "up")
    reordered = store.load_or_create(output)

    assert [card["card_id"] for card in reordered["cards"]] == ["manual_0001", "c_1"]


def test_image_action_returns_only_media_fragment_without_page_reload(tmp_path):
    output = tmp_path / "output" / "video"
    output.mkdir(parents=True, exist_ok=True)
    project(output)
    store = ReviewStore()
    store.load_or_create(output)
    state = JobState(
        job_id="job1", source="Lesson", video_path=str(tmp_path / "video.mkv"),
        output_directory=str(output), status="awaiting_review", created_at="now",
    )
    manager = JobManager(output_root=tmp_path / "output")
    manager.jobs[state.job_id] = state
    client = TestClient(create_app(root=tmp_path, manager=manager))
    token = client.get("/").cookies["anki_csrf"]

    response = client.post(
        "/jobs/job1/review/cards/c_1/image-remove",
        data={"csrf": token},
        headers={"HX-Request": "true"},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert 'id="media-c_1"' in response.text
    assert "<html" not in response.text
