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

