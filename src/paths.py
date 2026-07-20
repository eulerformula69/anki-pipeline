from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class ResultPaths:
    directory: Path
    audio: Path
    transcript_json: Path
    transcript_txt: Path
    semantic_segments_json: Path
    semantic_diagnostic_json: Path
    knowledge_items_json: Path
    knowledge_diagnostic_json: Path
    knowledge_review_json: Path
    review_diagnostic_json: Path
    card_candidates_json: Path
    cards_diagnostic_json: Path
    target_expressions_json: Path
    target_expressions_diagnostic_json: Path
    target_transcript_json: Path
    target_audio_directory: Path
    card_media_json: Path
    media_directory: Path
    anki_deck: Path
    anki_export_manifest_json: Path
    approved_cards_json: Path
    gui_job_json: Path
    gui_log: Path
    review_media_directory: Path
    media_candidates_directory: Path


def source_manifest(video: Path) -> dict[str, object] | None:
    path = video.parent / "source_manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def project_key(video: Path) -> str:
    manifest = source_manifest(video)
    source_id = manifest and manifest.get("source_id")
    if isinstance(source_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{3,80}", source_id):
        return source_id
    return video.stem


def source_identity(video: Path) -> str:
    manifest = source_manifest(video)
    source_id = manifest and manifest.get("source_id")
    if isinstance(source_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{3,80}", source_id):
        return f"youtube:{source_id}"
    return video.name


def build_result_paths(video: Path, output_root: Path = Path("output")) -> ResultPaths:
    key = project_key(video)
    directory = output_root / key
    return ResultPaths(
        directory=directory,
        audio=directory / "audio.wav",
        transcript_json=directory / "transcript.json",
        transcript_txt=directory / "transcript.txt",
        semantic_segments_json=directory / "semantic_segments.json",
        semantic_diagnostic_json=directory / "semantic_segments_diagnostic.json",
        knowledge_items_json=directory / "knowledge_items.json",
        knowledge_diagnostic_json=directory / "knowledge_items_diagnostic.json",
        knowledge_review_json=directory / "knowledge_review.json",
        review_diagnostic_json=directory / "knowledge_review_diagnostic.json",
        card_candidates_json=directory / "card_candidates.json",
        cards_diagnostic_json=directory / "card_candidates_diagnostic.json",
        target_expressions_json=directory / "target_expressions.json",
        target_expressions_diagnostic_json=directory / "target_expressions_diagnostic.json",
        target_transcript_json=directory / "target_transcript.json",
        target_audio_directory=directory / "target_audio",
        card_media_json=directory / "card_media.json",
        media_directory=directory / "media",
        anki_deck=directory / f"{key}.apkg",
        anki_export_manifest_json=directory / "anki_export_manifest.json",
        approved_cards_json=directory / "approved_cards.json",
        gui_job_json=directory / "gui_job.json",
        gui_log=directory / "gui.log",
        review_media_directory=directory / "review_media",
        media_candidates_directory=directory / "media_candidates",
    )
