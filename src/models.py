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


@dataclass(frozen=True)
class DuplicateOccurrence:
    semantic_segment_id: int
    source_transcript_segment_ids: list[int]


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    semantic_segment_id: int
    type: str
    statement: str
    explanation: str
    importance: int
    source_transcript_segment_ids: list[int]
    evidence: str
    duplicate_occurrences: list[DuplicateOccurrence] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.duplicate_occurrences is None:
            data.pop("duplicate_occurrences")
        return data


@dataclass(frozen=True)
class KnowledgeCollection:
    source: str
    model: str
    knowledge_items: list[KnowledgeItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "knowledge_items": [item.to_dict() for item in self.knowledge_items],
        }
