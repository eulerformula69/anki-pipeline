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


@dataclass(frozen=True)
class KnowledgeReview:
    knowledge_id: str
    status: str
    confidence: float
    issues: list[str]
    recommended_action: str
    reason: str


@dataclass(frozen=True)
class KnowledgeReviewCollection:
    source: str
    model: str
    reviews: list[KnowledgeReview]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CardCandidate:
    card_id: str
    knowledge_ids: list[str]
    card_type: str
    front: str
    back: str
    extra: str
    difficulty: int
    importance: int
    tags: list[str]
    source_transcript_segment_ids: list[int]
    expression_ids: list[str] | None = None


@dataclass(frozen=True)
class CardCandidateCollection:
    source: str
    model: str
    card_candidates: list[CardCandidate]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetExpression:
    id: str
    language: str
    surface: str
    reading: str
    romanization: str
    meaning: str
    knowledge_ids: list[str]
    source_transcript_segment_ids: list[int]
    verification: str
    confidence: float
    type: str = "term"


@dataclass(frozen=True)
class TargetExpressionCollection:
    source: str
    model: str
    target_language: str
    expressions: list[TargetExpression]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetTranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    source_transcript_segment_ids: list[int]
    language_probability: float


@dataclass(frozen=True)
class TargetTranscript:
    source: str
    language: str
    duration_seconds: float
    model: str
    segments: list[TargetTranscriptSegment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameMetrics:
    brightness_mean: float
    dark_ratio: float
    sharpness: float


@dataclass(frozen=True)
class CardImage:
    path: str
    timestamp: float
    width: int
    height: int
    selection_method: str
    quality_score: float
    metrics: FrameMetrics


@dataclass(frozen=True)
class CardMedia:
    card_id: str
    source_transcript_segment_ids: list[int]
    start: float
    end: float
    image: CardImage | None
    selection_method: str | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.image is not None:
            data.pop("selection_method")
            data.pop("rejection_reason")
        return data


@dataclass(frozen=True)
class CardMediaCollection:
    source: str
    generator: str
    frames_per_card: int
    image_policy: str
    cards: list[CardMedia]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "generator": self.generator,
            "frames_per_card": self.frames_per_card,
            "image_policy": self.image_policy,
            "cards": [card.to_dict() for card in self.cards],
        }
