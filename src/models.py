from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    source: str
    language: str
    duration_seconds: float
    model: str
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticSegment:
    id: int
    title: str
    start: float
    end: float
    summary: str
    transcript_segment_ids: list[int]
    text: str


@dataclass(frozen=True)
class SemanticTranscript:
    source: str
    model: str
    segments: list[SemanticSegment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
