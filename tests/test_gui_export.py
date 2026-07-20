import json

import pytest

from src.anki_export import AnkiExportError, build_anki_deck


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
