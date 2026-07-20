from pathlib import Path

from src.gui.services import compute_plan


def test_plan_adds_missing_dependencies(tmp_path):
    assert compute_plan(["frame_extraction"], tmp_path) == [
        "transcription", "semantic_segmentation", "knowledge_extraction",
        "knowledge_review", "card_generation", "frame_extraction",
    ]


def test_plan_reuses_existing_artifacts(tmp_path):
    for name in ("transcript.json", "semantic_segments.json", "knowledge_items.json", "knowledge_review.json", "card_candidates.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert compute_plan(["frame_extraction"], tmp_path) == ["frame_extraction"]
