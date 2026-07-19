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


def build_result_paths(video: Path, output_root: Path = Path("output")) -> ResultPaths:
    directory = output_root / video.stem
    return ResultPaths(
        directory=directory,
        audio=directory / "audio.wav",
        transcript_json=directory / "transcript.json",
        transcript_txt=directory / "transcript.txt",
        semantic_segments_json=directory / "semantic_segments.json",
        semantic_diagnostic_json=directory / "semantic_segments_diagnostic.json",
    )
