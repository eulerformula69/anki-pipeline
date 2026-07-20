import json

import pytest

from src.gui.review_store import ReviewError, ReviewStore


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
