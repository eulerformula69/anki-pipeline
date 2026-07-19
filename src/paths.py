from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


def build_result_paths(video: Path, output_root: Path = Path("output")) -> ResultPaths:
    directory = output_root / video.stem
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
    )
