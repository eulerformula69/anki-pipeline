import json

import pytest

from src.models import KnowledgeItem, KnowledgeReview
from src.review import (
    KnowledgeReviewError,
    group_knowledge_items,
    normalize_review_relations,
    parse_review_response,
    review_knowledge,
)


def items() -> list[KnowledgeItem]:
    return [
        KnowledgeItem("k_0001", 0, "fact", "Alpha is useful.", "", 4, [1], "Alpha."),
        KnowledgeItem("k_0002", 0, "example", "Alpha example.", "", 2, [2], "Example."),
        KnowledgeItem("k_0003", 1, "fact", "Beta is useful.", "", 3, [3], "Beta."),
    ]


def review_item(knowledge_id: str, status: str = "keep", **overrides):
    value = {
        "knowledge_id": knowledge_id,
        "status": status,
        "confidence": 0.9,
        "issues": [],
        "recommended_action": "none",
        "reason": "The item is suitable.",
    }
    value.update(overrides)
    return value


def test_keep_discard_merge_and_split_are_valid() -> None:
    data = {
        "reviews": [
            review_item("k1", "keep"),
            review_item("k2", "discard", recommended_action="discard"),
            review_item("k3", "merge", issues=["duplicate"], recommended_action="merge"),
            review_item("k4", "split", issues=["contains_two_facts"], recommended_action="split"),
        ]
    }

    result = parse_review_response(data, ["k1", "k2", "k3", "k4"])

    assert [item.status for item in result] == ["keep", "discard", "merge", "split"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "unknown", "status"),
        ("issues", ["invented"], "issues"),
        ("confidence", 5.1, "confidence"),
        ("recommended_action", "invented", "recommended_action"),
    ],
)
def test_invalid_review_values(field, value, message) -> None:
    data = {"reviews": [review_item("k1", **{field: value})]}

    with pytest.raises(KnowledgeReviewError, match=message):
        parse_review_response(data, ["k1"])


def test_duplicate_or_missing_review_id_is_invalid() -> None:
    data = {"reviews": [review_item("k1"), review_item("k1")]}

    with pytest.raises(KnowledgeReviewError, match="покрытие"):
        parse_review_response(data, ["k1", "k2"])


def test_five_point_confidence_is_normalized() -> None:
    data = {"reviews": [review_item("k1", confidence=5)]}

    result = parse_review_response(data, ["k1"])

    assert result[0].confidence == 1.0


def test_small_collection_stays_in_one_review_group() -> None:
    groups = group_knowledge_items(items())

    assert [[item.id for item in group] for group in groups] == [
        ["k_0001", "k_0002", "k_0003"],
    ]


def test_large_collection_is_split_below_common_response_limit() -> None:
    many_items = [
        KnowledgeItem(
            f"k_{index:04d}", 0, "fact", f"Fact {index}.", "", 3, [index], "Evidence."
        )
        for index in range(1, 41)
    ]

    groups = group_knowledge_items(many_items)

    assert [len(group) for group in groups] == [12, 12, 12, 4]
    assert [item.id for group in groups for item in group] == [
        item.id for item in many_items
    ]


def test_merge_into_discarded_duplicate_becomes_keep() -> None:
    reviews = [
        KnowledgeReview("k_0001", "merge", 0.9, ["duplicate"], "merge", "Merge with k_0002"),
        KnowledgeReview("k_0002", "discard", 0.9, ["redundant"], "discard", "Duplicate of k_0001"),
    ]

    result = normalize_review_relations(reviews)

    assert result[0].status == "keep"
    assert result[0].recommended_action == "none"
    assert result[1].status == "discard"


class SequenceClient:
    model = "qwen3:8b"

    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def generate_json(self, prompt: str):
        self.calls += 1
        return next(self.responses)


def write_knowledge(tmp_path) -> object:
    path = tmp_path / "knowledge.json"
    path.write_text(
        json.dumps(
            {
                "source": "video.mkv",
                "model": "qwen3:8b",
                "knowledge_items": [item.to_dict() for item in items()[:2]],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_review_retries_and_saves_utf8(tmp_path) -> None:
    invalid = ({"reviews": []}, "invalid")
    corrected_data = {
        "reviews": [
            review_item("k_0001", reason="Полезное знание."),
            review_item("k_0002", "discard", recommended_action="discard"),
        ]
    }
    client = SequenceClient([invalid, (corrected_data, "valid")])
    destination = tmp_path / "review.json"

    result = review_knowledge(
        write_knowledge(tmp_path), destination, tmp_path / "diagnostic.json", client
    )

    assert client.calls == 2
    assert result.reviews[0].reason == "Полезное знание."
    assert "Полезное" in destination.read_text(encoding="utf-8")


def test_invalid_review_uses_keep_fallback_and_preserves_diagnostic(tmp_path) -> None:
    invalid = ({"reviews": []}, "invalid")
    client = SequenceClient([invalid, invalid])
    destination = tmp_path / "review.json"
    destination.write_text("existing", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic.json"

    result = review_knowledge(write_knowledge(tmp_path), destination, diagnostic, client)

    assert [item.status for item in result.reviews] == ["keep", "keep"]
    assert all(item.confidence == 0 for item in result.reviews)
    assert len(json.loads(diagnostic.read_text(encoding="utf-8"))["ollama_responses"]) == 2
