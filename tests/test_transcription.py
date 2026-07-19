from dataclasses import dataclass

from src.transcription import collect_segments


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str


def test_collect_segments_preserves_timestamps() -> None:
    segments = collect_segments(
        [
            FakeSegment(start=0.0, end=2.5, text=" Hello "),
            FakeSegment(start=2.5, end=5.0, text="world"),
        ],
        duration=5.0,
    )

    assert [segment.id for segment in segments] == [0, 1]
    assert segments[0].end == 2.5
    assert segments[0].text == "Hello"
