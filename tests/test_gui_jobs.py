import json
from dataclasses import asdict
from pathlib import Path

from src.gui.jobs import JobManager
from src.gui.schemas import JobState, PipelineRunRequest
from src.gui.services import (
    PipelineRunner, artifact_is_stale, compute_plan, migrate_legacy_project,
)


def test_plan_adds_missing_dependencies(tmp_path):
    assert compute_plan(["frame_extraction"], tmp_path) == [
        "transcription", "target_expressions", "target_transcription",
        "semantic_segmentation", "knowledge_extraction",
        "knowledge_review", "card_generation", "frame_extraction",
    ]


def test_plan_reuses_existing_artifacts(tmp_path):
    for name in ("transcript.json", "semantic_segments.json", "knowledge_items.json", "knowledge_review.json", "card_candidates.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert compute_plan(["frame_extraction"], tmp_path) == ["frame_extraction"]


def test_artifact_is_stale_when_language_dependency_is_newer(tmp_path):
    semantic = tmp_path / "semantic_segments.json"
    target = tmp_path / "target_transcript.json"
    semantic.write_text("{}", encoding="utf-8")
    target.write_text("{}", encoding="utf-8")
    semantic.touch()
    assert artifact_is_stale("semantic_segmentation", tmp_path) is False
    target.touch()
    target_stat = target.stat()
    target.touch()
    assert target.stat().st_mtime_ns >= target_stat.st_mtime_ns
    if target.stat().st_mtime_ns == semantic.stat().st_mtime_ns:
        target.write_text('{"new":true}', encoding="utf-8")
    assert artifact_is_stale("semantic_segmentation", tmp_path) is True


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


def test_target_transcription_refreshes_expressions_with_native_evidence(tmp_path, monkeypatch):
    video = tmp_path / "video.mkv"
    video.write_bytes(b"video")
    output = tmp_path / "output" / "video"
    output.mkdir(parents=True)
    (output / "transcript.json").write_text("{}", encoding="utf-8")
    (output / "target_expressions.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.gui.services.run_pipeline",
        lambda path, **kwargs: calls.append(kwargs) or path,
    )

    PipelineRunner().run(
        PipelineRunRequest(video, ["target_transcription"], "Lesson"),
        lambda stage, event: None,
    )

    assert calls[0]["transcribe_target_stage"] is True
    assert calls[1]["extract_expressions"] is True
    assert calls[1]["force"] is True


def test_legacy_youtube_output_is_copied_to_unique_project(tmp_path):
    video = tmp_path / "input" / "downloads" / "Youtube" / "abc123" / "video.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    (video.parent / "source_manifest.json").write_text(
        '{"source_id":"abc123"}', encoding="utf-8"
    )
    legacy = tmp_path / "output" / "video"
    legacy.mkdir(parents=True)
    (legacy / "gui_job.json").write_text(json.dumps({
        "video_path": str(video.resolve()), "output_directory": str(legacy.resolve()),
    }), encoding="utf-8")
    (legacy / "transcript.json").write_text(
        '{"source":"video.mkv","segments":[]}', encoding="utf-8"
    )

    destination = migrate_legacy_project(video, tmp_path / "output")

    assert destination == tmp_path / "output" / "abc123"
    migrated = json.loads((destination / "transcript.json").read_text(encoding="utf-8"))
    assert migrated["source"] == "youtube:abc123"
    assert json.loads((legacy / "transcript.json").read_text(encoding="utf-8"))["source"] == "video.mkv"
