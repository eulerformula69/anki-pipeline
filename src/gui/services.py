from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ..anki_export import build_anki_deck
from ..frame_extraction import calculate_frame_metrics, calculate_quality_score, choose_frame_timestamps, extract_frame
from ..semantic import save_json_atomic
from ..paths import build_result_paths, source_identity
from ..pipeline import run_pipeline
from .schemas import PipelineRunRequest, STAGES


ARTIFACTS = {
    "transcription": "transcript.json", "semantic_segmentation": "semantic_segments.json",
    "knowledge_extraction": "knowledge_items.json", "knowledge_review": "knowledge_review.json",
    "target_expressions": "target_expressions.json", "target_transcription": "target_transcript.json",
    "card_generation": "card_candidates.json", "frame_extraction": "card_media.json",
}
DEPENDENCIES = {
    "semantic_segmentation": ("transcription",),
    "knowledge_extraction": ("semantic_segmentation",),
    "knowledge_review": ("knowledge_extraction",),
    "target_expressions": ("knowledge_extraction",),
    "target_transcription": ("transcription", "target_expressions"),
    "card_generation": ("knowledge_extraction", "knowledge_review"),
    "frame_extraction": ("transcription", "card_generation"),
}


def migrate_legacy_project(video: Path, output_root: Path) -> Path:
    destination = build_result_paths(video, output_root).directory
    legacy = output_root / video.stem
    if destination == legacy or destination.exists() or not legacy.is_dir():
        return destination
    state_path = legacy / "gui_job.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        recorded_video = Path(state["video_path"]).resolve()
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return destination
    if recorded_video != video.resolve():
        return destination
    shutil.copytree(legacy, destination)
    identity = source_identity(video)
    for path in destination.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        changed = False
        if "source" in data:
            data["source"] = identity
            changed = True
        if path.name == "gui_job.json":
            data["output_directory"] = str(destination.resolve())
            changed = True
        if changed:
            save_json_atomic(data, path)
    return destination


def compute_plan(selected: list[str], output: Path) -> list[str]:
    unknown = set(selected) - set(STAGES)
    if unknown:
        raise ValueError(f"Неизвестные стадии: {', '.join(sorted(unknown))}")
    needed = set(selected)
    changed = True
    while changed:
        changed = False
        for stage in tuple(needed):
            for dependency in DEPENDENCIES.get(stage, ()):
                if not (output / ARTIFACTS[dependency]).exists() and dependency not in needed:
                    needed.add(dependency)
                    changed = True
    return [stage for stage in STAGES if stage in needed]


class PipelineRunner:
    def run(self, request: PipelineRunRequest, progress: Callable[[str, str], None]) -> Path:
        paths = build_result_paths(request.video_path)
        plan = compute_plan(request.stages, paths.directory)
        common = dict(
            model=request.model, device=request.device, language=request.language,
            force=request.force, ollama_model=request.ollama_model,
            ollama_url=request.ollama_url, ollama_timeout=request.ollama_timeout,
            knowledge_language=request.knowledge_language,
            knowledge_min_importance=request.knowledge_min_importance,
            knowledge_max_items_per_segment=request.knowledge_max_items_per_segment,
            target_language=request.target_language, target_glossary=request.target_glossary,
            target_whisper_model=request.target_whisper_model,
            frames_per_card=request.frames_per_card, image_policy=request.image_policy,
        )
        result = request.video_path
        for stage in plan:
            progress(stage, "started")
            flags = {
                "semantic_segmentation": {"semantic_segment": True},
                "knowledge_extraction": {"extract_knowledge_items": True},
                "knowledge_review": {"review_knowledge_items": True},
                "target_expressions": {"extract_expressions": True},
                "target_transcription": {"transcribe_target_stage": True},
                "card_generation": {"generate_cards": True},
                "frame_extraction": {"extract_card_images_stage": True},
            }.get(stage, {})
            result = run_pipeline(request.video_path, **common, **flags)
            if stage == "target_transcription":
                progress("target_expressions", "started")
                refreshed = {**common, "force": True}
                result = run_pipeline(
                    request.video_path, **refreshed, extract_expressions=True
                )
                progress("target_expressions", "completed")
            progress(stage, "completed")
        return result


def export_approved_deck(
    video: Path,
    deck_name: str | None = None,
    global_tags: list[str] | None = None,
) -> Path:
    paths = build_result_paths(video)
    build_anki_deck(
        paths.card_candidates_json, paths.target_expressions_json,
        paths.card_media_json, paths.anki_deck, paths.anki_export_manifest_json,
        deck_name=deck_name, approved_cards_path=paths.approved_cards_json,
        global_tags=global_tags,
    )
    return paths.anki_deck


def generate_image_candidates(video: Path, output: Path, card_id: str, count: int = 5) -> list[dict[str, object]]:
    data = json.loads((output / "card_media.json").read_text(encoding="utf-8"))
    item = next((card for card in data.get("cards", []) if card.get("card_id") == card_id), None)
    if item is None:
        raise ValueError("Для карточки не найден временной диапазон.")
    start, end = float(item["start"]), float(item["end"])
    directory = output / "media_candidates" / card_id
    directory.mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, timestamp in enumerate(choose_frame_timestamps(start, end, count), 1):
        path = directory / f"frame_{index:03d}.jpg"
        extract_frame(video, timestamp, path)
        metrics = calculate_frame_metrics(path)
        candidates.append({
            "path": path.relative_to(output).as_posix(), "timestamp": timestamp,
            "quality_score": calculate_quality_score(metrics), "metrics": asdict(metrics),
        })
    save_json_atomic({"card_id": card_id, "candidates": candidates}, directory / "candidates.json")
    return candidates
