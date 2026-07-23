import json

import pytest

from src.models import KnowledgeItem, TranscriptSegment
from src.target_expressions import (
    TargetExpressionError,
    extract_target_expressions,
    glossary_expressions_from_transcript,
    load_glossary,
    parse_response,
    recover_partial_expressions,
    _normalize_native,
    stable_expression_id,
)
from src.ollama_schemas import target_expression_response_schema


def item() -> KnowledgeItem:
    return KnowledgeItem(
        "k_0001", 0, "definition", "Datte can introduce a reason.",
        "Japanese grammar", 5, [3, 4], "Datte...",
    )


def valid_response():
    return {
        "expressions": [{
            "type": "term",
            "surface": "だって", "reading": "だって", "romanization": "datte",
            "meaning": "Introduces a reason",
            "source_transcript_segment_ids": [3, 4], "confidence": 0.9,
        }]
    }


def test_parse_expression_keeps_native_writing_and_links() -> None:
    result = parse_response(valid_response(), [item()], "ja")

    assert result[0].surface == "だって"
    assert result[0].verification == "needs_review"
    assert result[0].knowledge_ids == ["k_0001"]


def test_early_expression_does_not_require_knowledge_items() -> None:
    result = parse_response(
        valid_response(), [], "ja", allowed_source_ids={3, 4}
    )
    assert result[0].knowledge_ids == []
    assert result[0].source_transcript_segment_ids == [3, 4]


def test_unknown_source_id_is_rejected() -> None:
    data = valid_response()
    data["expressions"][0]["source_transcript_segment_ids"] = [99]

    with pytest.raises(TargetExpressionError, match="source_transcript_segment_ids"):
        parse_response(data, [item()], "ja")


def test_japanese_romanized_surface_is_rejected() -> None:
    data = valid_response()
    data["expressions"][0]["surface"] = "datte"

    with pytest.raises(TargetExpressionError, match="письменность"):
        parse_response(data, [item()], "ja")


def test_mixed_latin_and_japanese_surface_is_rejected() -> None:
    data = valid_response()
    data["expressions"][0]["surface"] = "私 datte"

    with pytest.raises(TargetExpressionError, match="письменность"):
        parse_response(data, [item()], "ja")


def test_glossary_replaces_romaji_and_verifies_expression(tmp_path) -> None:
    path = tmp_path / "ja.json"
    path.write_text(json.dumps({"entries": [{
        "romanization": "datte", "surface": "だって", "reading": "だって"
    }]}), encoding="utf-8")
    data = valid_response()
    data["expressions"][0]["surface"] = "datte"
    data["expressions"][0]["reading"] = "datte"

    result = parse_response(data, [item()], "ja", load_glossary(path))

    assert result[0].surface == "だって"
    assert result[0].verification == "verified"


def test_second_pass_native_evidence_marks_expression_transcribed() -> None:
    result = parse_response(
        valid_response(), [item()], "ja", native_evidence={3: ["だって。"]},
        primary_evidence={3: "The particle datte is used here."},
    )

    assert result[0].verification == "transcribed"


def test_second_pass_without_primary_phonetic_match_is_not_trusted() -> None:
    result = parse_response(
        valid_response(), [item()], "ja", native_evidence={3: ["だって。"]},
        primary_evidence={3: "This is an unrelated English explanation."},
    )

    assert result[0].verification == "needs_review"


class Client:
    model = "qwen3:8b"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate_json(self, prompt):
        self.calls += 1
        return next(self.responses)


def test_extraction_retries_without_overwriting_existing_result(tmp_path) -> None:
    knowledge = tmp_path / "knowledge.json"
    knowledge.write_text(json.dumps({
        "source": "video.mkv", "model": "qwen3:8b",
        "knowledge_items": [item().to_dict()],
    }), encoding="utf-8")
    destination = tmp_path / "expressions.json"
    diagnostic = tmp_path / "diagnostic.json"
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"segments": [
        {"id": 3, "start": 1.0, "end": 2.0, "text": "datte"},
        {"id": 4, "start": 2.0, "end": 3.0, "text": "Japanese grammar"},
    ]}), encoding="utf-8")
    invalid = ({"expressions": [{"surface": "だって"}]}, "bad")
    client = Client([invalid, (valid_response(), "good")])

    result = extract_target_expressions(
        transcript, knowledge, destination, diagnostic, client, target_language="ja"
    )

    assert client.calls == 2
    assert result.expressions[0].id == stable_expression_id("ja", "term", "だって")
    assert json.loads(destination.read_text(encoding="utf-8"))["target_language"] == "ja"


def test_extraction_skips_window_with_only_non_native_hallucinations(tmp_path) -> None:
    knowledge = tmp_path / "knowledge.json"
    knowledge.write_text(json.dumps({
        "source": "video.mkv", "model": "qwen3:8b",
        "knowledge_items": [item().to_dict()],
    }), encoding="utf-8")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"segments": [
        {"id": 3, "start": 1.0, "end": 2.0, "text": "Class dismissed."},
    ]}), encoding="utf-8")
    invalid = valid_response()
    invalid["expressions"][0].update(
        surface="Class dismissed", reading="クラスディスミッスド"
    )
    client = Client([(invalid, "first"), (invalid, "second")])

    result = extract_target_expressions(
        transcript, knowledge, tmp_path / "expressions.json",
        tmp_path / "diagnostic.json", client, target_language="ja",
    )

    assert result.expressions == []
    assert client.calls == 2
    assert not (tmp_path / "diagnostic.json").exists()


def test_expression_id_is_stable_and_content_specific() -> None:
    first = stable_expression_id("ja", "term", "だって")

    assert first == stable_expression_id("ja", "term", "だって。")
    assert first != stable_expression_id("ja", "example_sentence", "だって")


def test_schema_caps_expression_count() -> None:
    assert target_expression_response_schema()["properties"]["expressions"]["maxItems"] == 6


def test_complete_expressions_are_recovered_from_truncated_json() -> None:
    complete = valid_response()["expressions"][0]
    raw = json.dumps({"expressions": [complete]}, ensure_ascii=False)[:-2] + ', {"type":'

    recovered = recover_partial_expressions(raw)

    assert recovered == {"expressions": [complete]}


def test_punctuation_variants_share_a_stable_id() -> None:
    plain = "\u3060\u3063\u3066"
    decorated = "\u301c\u3060\u3063\u3066\u3002"

    assert stable_expression_id("ja", "term", plain) == stable_expression_id(
        "ja", "term", decorated
    )


def test_wave_dash_does_not_block_native_evidence_match() -> None:
    assert _normalize_native("〜らしい") == _normalize_native("らしい")


def test_glossary_recovers_known_expression_without_llm_output() -> None:
    segments = [
        TranscriptSegment(
            id=7,
            start=1.0,
            end=2.0,
            text="Yoda is used for a simile.",
        )
    ]
    glossary = {
        "yoda": {
            "surface": "ようだ",
            "reading": "ようだ",
            "meaning": "similar to",
            "type": "grammar_pattern",
            "aliases": ["yohda"],
        }
    }

    result = glossary_expressions_from_transcript(
        segments,
        glossary,
        [],
        language="ja",
    )

    assert len(result) == 1
    assert result[0].surface == "ようだ"
    assert result[0].verification == "verified"
    assert result[0].source_transcript_segment_ids == [7]
    assert result[0].romanization_aliases == ["yohda"]


def test_glossary_does_not_match_inside_an_english_word() -> None:
    segments = [
        TranscriptSegment(
            id=7,
            start=1.0,
            end=2.0,
            text="You can compare these forms.",
        )
    ]
    glossary = {
        "ka": {
            "surface": "か",
            "reading": "か",
            "type": "grammar_pattern",
        }
    }

    result = glossary_expressions_from_transcript(
        segments,
        glossary,
        [],
        language="ja",
    )

    assert result == []
