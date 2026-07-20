import json
from dataclasses import asdict
from pathlib import Path

from src.gui.jobs import JobManager
from src.gui.schemas import JobState
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


def test_manager_restores_saved_job_after_restart(tmp_path):
    output = tmp_path / "output" / "video"
    output.mkdir(parents=True)
    state = JobState(
        job_id="saved-job", source="Lesson", video_path=str(tmp_path / "video.mkv"),
        output_directory=str(output), status="failed", created_at="2026-07-20T00:00:00Z",
    )
    (output / "gui_job.json").write_text(json.dumps(asdict(state)), encoding="utf-8")

    manager = JobManager(output_root=tmp_path / "output")

    assert manager.get("saved-job").source == "Lesson"
