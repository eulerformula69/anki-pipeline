import json

import pytest

from src.models import TranscriptSegment
from src.target_transcription import (
    TargetTranscriptionError,
    build_target_ranges,
    load_candidate_source_ids,
)


def segments():
    return [
        TranscriptSegment(0, 0.0, 4.0, "English"),
        TranscriptSegment(1, 4.0, 8.0, "ashita wa hare datte"),
        TranscriptSegment(2, 8.0, 12.0, "Explanation"),
        TranscriptSegment(3, 12.5, 16.5, "watashi datte dekiru"),
    ]


def test_target_ranges_merge_consecutive_segments_and_keep_ids() -> None:
    result = build_target_ranges(
        segments(), {1, 2, 3}, duration=20.0, padding=0.5
    )

    assert result[0].source_transcript_segment_ids == [1, 2, 3]
    assert result[0].start == 3.5
    assert result[0].end == 17.0


def test_target_ranges_respect_maximum_duration() -> None:
    result = build_target_ranges(
        segments(), {1, 2, 3}, duration=24.0, padding=0.0,
        max_range_seconds=10.0,
    )

    assert [item.source_transcript_segment_ids for item in result] == [[1, 2], [3]]


def test_target_ranges_require_candidates() -> None:
    with pytest.raises(TargetTranscriptionError, match="нет таймкодов"):
        build_target_ranges(segments(), set(), duration=24.0)


def test_candidate_ids_are_loaded_without_duplicates(tmp_path) -> None:
    path = tmp_path / "expressions.json"
    path.write_text(json.dumps({"expressions": [
        {"source_transcript_segment_ids": [1, 2]},
        {"source_transcript_segment_ids": [2, 3]},
    ]}), encoding="utf-8")

    assert load_candidate_source_ids(path) == {1, 2, 3}


def test_early_unverified_expression_is_sent_to_second_pass(tmp_path) -> None:
    path = tmp_path / "expressions.json"
    path.write_text(json.dumps({"expressions": [{
        "type": "grammar_pattern", "verification": "needs_review",
        "romanization": "ashita wa hare datte",
        "source_transcript_segment_ids": [1],
    }]}), encoding="utf-8")
    primary = {1: "The example is ashita wa hare datte"}
    assert load_candidate_source_ids(path, primary) == {1}
