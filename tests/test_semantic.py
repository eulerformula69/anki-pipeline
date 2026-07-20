import json

import pytest

from src.models import TranscriptSegment
from src.semantic import (
    ProposedBlock,
    SemanticSegmentationError,
    assemble_result,
    build_prompt,
    build_windows,
    parse_proposals,
    semantic_segment_transcript,
)


def transcript_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(id=0, start=0.0, end=2.0, text="First sentence."),
        TranscriptSegment(id=1, start=2.0, end=5.5, text="Second sentence."),
        TranscriptSegment(id=2, start=5.5, end=8.0, text="Third sentence."),
    ]


def test_missing_ids_are_rejected() -> None:
    data = {
        "segments": [
            {"title": "Тема", "summary": "Описание", "transcript_segment_ids": [0, 2]}
        ]
    }

    with pytest.raises(SemanticSegmentationError, match="покрытие ID"):
        parse_proposals(data, [0, 1, 2])


def test_duplicate_ids_are_rejected() -> None:
    data = {
        "segments": [
            {
                "title": "Тема",
                "summary": "Описание",
                "transcript_segment_ids": [0, 1, 1, 2],
            }
        ]
    }

    with pytest.raises(SemanticSegmentationError, match="покрытие ID"):
        parse_proposals(data, [0, 1, 2])


def test_assemble_text_and_timestamps_from_original_segments() -> None:
    result = assemble_result(
        "video.webm",
        "qwen3:8b",
        transcript_segments(),
        [
            ProposedBlock("Тема", "Описание", [0, 1]),
            ProposedBlock("Другая", "Продолжение", [2]),
        ],
    )

    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 5.5
    assert result.segments[0].text == "First sentence. Second sentence."


def test_assemble_preserves_target_language_text() -> None:
    result = assemble_result(
        "video.webm", "gemini", transcript_segments(),
        [ProposedBlock("Topic", "Summary", [0, 1, 2])],
        {1: ["明日は晴れだって"]},
    )
    assert result.segments[0].target_language_text == "明日は晴れだって"


def test_windows_keep_overlap_outside_core() -> None:
    windows = build_windows(transcript_segments(), max_chars=50, overlap=1)

    assert len(windows) > 1
    assert windows[0].core_ids == [0]
    assert [segment.id for segment in windows[1].context][:2] == [0, 1]


def test_semantic_prompt_uses_selected_card_language() -> None:
    window = build_windows(transcript_segments())[0]
    assert "Write title and summary in English" in build_prompt(
        window, output_language="en"
    )
    assert "Write title and summary in Russian" in build_prompt(
        window, output_language="ru"
    )


class RetryClient:
    model = "qwen3:8b"

    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str):
        self.calls += 1
        if self.calls == 1:
            return {"segments": []}, '{"segments":[]}'
        return (
            {
                "segments": [
                    {
                        "title": "Тема",
                        "summary": "Описание",
                        "transcript_segment_ids": [0, 1, 2],
                    }
                ]
            },
            "valid",
        )


def test_invalid_result_is_retried_once(tmp_path) -> None:
    transcript = {
        "source": "video.webm",
        "segments": [segment.__dict__ for segment in transcript_segments()],
    }
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    client = RetryClient()

    result = semantic_segment_transcript(
        transcript_path,
        tmp_path / "semantic.json",
        tmp_path / "diagnostic.json",
        client,
    )

    assert client.calls == 2
    assert result.segments[0].transcript_segment_ids == [0, 1, 2]


class AlwaysInvalidClient:
    model = "qwen3:8b"

    def generate_json(self, prompt: str):
        return {"segments": []}, '{"segments":[]}'


def test_two_invalid_results_create_diagnostic_without_overwriting(tmp_path) -> None:
    transcript = {
        "source": "video.webm",
        "segments": [segment.__dict__ for segment in transcript_segments()],
    }
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    destination = tmp_path / "semantic.json"
    destination.write_text("existing result", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic.json"

    with pytest.raises(SemanticSegmentationError, match="дважды"):
        semantic_segment_transcript(
            transcript_path,
            destination,
            diagnostic,
            AlwaysInvalidClient(),
        )

    assert destination.read_text(encoding="utf-8") == "existing result"
    assert len(json.loads(diagnostic.read_text(encoding="utf-8"))["ollama_responses"]) == 2
