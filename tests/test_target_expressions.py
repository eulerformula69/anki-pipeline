import json

import pytest

from src.models import KnowledgeItem
from src.target_expressions import (
    TargetExpressionError,
    extract_target_expressions,
    load_glossary,
    parse_response,
)


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
    assert result.expressions[0].id == "e_0001"
    assert json.loads(destination.read_text(encoding="utf-8"))["target_language"] == "ja"
