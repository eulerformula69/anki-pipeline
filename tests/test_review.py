import json

import numpy as np
import pytest

from src.models import KnowledgeItem, KnowledgeReview
from src.review import (
    KnowledgeReviewError,
    build_review_prompt,
    group_knowledge_items,
    normalize_review_relations,
    parse_review_response,
    review_knowledge,
)
from src.review_preferences import ReviewPreferenceExample, ReviewPreferenceHistory
from src.content_quality import enforce_content_quality


def test_review_prompt_discards_non_learning_video_content() -> None:
    prompt = build_review_prompt([])

    assert "use outside this specific video" in prompt
    assert "must receive status=discard" in prompt


def test_review_prompt_includes_personal_preferences_when_available() -> None:
    history = ReviewPreferenceHistory(
        examples=[
            ReviewPreferenceExample(True, "What does ばかり mean?", "Nothing but."),
            ReviewPreferenceExample(True, "How is ようだ used?", "For likeness."),
            ReviewPreferenceExample(True, "What does 別に mean?", "Not particularly."),
            ReviewPreferenceExample(False, "What is the channel plan?", "Weekly videos."),
            ReviewPreferenceExample(False, "Which game comes next?", "A voted game."),
            ReviewPreferenceExample(False, "What does the author like?", "Games."),
        ],
        approved_count=3,
        rejected_count=3,
        project_count=1,
    )

    prompt = build_review_prompt([], preference_history=history)

    assert "PERSONAL REVIEW PREFERENCES" in prompt
    assert "What does ばかり mean?" in prompt
    assert "Which game comes next?" in prompt


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


def test_extra_ids_from_another_group_are_ignored_and_expected_are_reordered() -> None:
    data = {
        "reviews": [
            review_item("k_old"),
            review_item("k2"),
            review_item("k1"),
        ]
    }

    result = parse_review_response(data, ["k1", "k2"])

    assert [item.knowledge_id for item in result] == ["k1", "k2"]


def test_duplicate_split_with_merge_action_is_normalized_to_merge() -> None:
    data = {
        "reviews": [
            review_item(
                "k1",
                "split",
                issues=["duplicate", "redundant"],
                recommended_action="merge",
                reason="Duplicates k2.",
            ),
            review_item("k2"),
        ]
    }

    result = parse_review_response(data, ["k1", "k2"])

    assert result[0].status == "merge"
    assert result[0].recommended_action == "merge"


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


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, statements: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[item] for item in statements], dtype=np.float32)


def test_semantic_grouping_puts_distant_paraphrases_together() -> None:
    many_items = [
        KnowledgeItem(
            f"k_{index:04d}", 0, "fact", f"Fact {index}.", "", 3, [index],
            "Evidence.",
        )
        for index in range(1, 14)
    ]
    vectors = {
        item.statement: [1.0, 0.0] if item.id in {"k_0001", "k_0013"}
        else [0.0, float(index + 1)]
        for index, item in enumerate(many_items)
    }

    groups = group_knowledge_items(
        many_items,
        max_group_size=12,
        embedder=FakeEmbedder(vectors),
        neighbor_threshold=0.95,
    )

    assert any(
        {"k_0001", "k_0013"} <= {item.id for item in group}
        for group in groups
    )
    assert all(len(group) <= 12 for group in groups)


def test_unrelated_semantic_components_are_packed_efficiently() -> None:
    many_items = [
        KnowledgeItem(
            f"k_{index:04d}", 0, "fact", f"Fact {index}.", "", 3, [index],
            "Evidence.",
        )
        for index in range(1, 14)
    ]
    vectors = {
        item.statement: [
            1.0 if dimension == index else 0.0
            for dimension in range(len(many_items))
        ]
        for index, item in enumerate(many_items)
    }

    groups = group_knowledge_items(
        many_items,
        max_group_size=12,
        embedder=FakeEmbedder(vectors),
        neighbor_threshold=0.95,
    )

    assert sorted(len(group) for group in groups) == [1, 12]


def test_review_output_restores_original_knowledge_order(
    tmp_path,
    monkeypatch,
) -> None:
    knowledge_path = write_knowledge(tmp_path)
    source = json.loads(knowledge_path.read_text(encoding="utf-8"))
    source["knowledge_items"].append(items()[2].to_dict())
    knowledge_path.write_text(json.dumps(source), encoding="utf-8")
    reordered = [[items()[2]], items()[:2]]
    monkeypatch.setattr("src.review.group_knowledge_items", lambda *args, **kwargs: reordered)
    responses = [
        ({"reviews": [review_item("k_0003")]}, "third"),
        ({
            "reviews": [
                review_item("k_0001"),
                review_item("k_0002"),
            ]
        }, "first two"),
    ]

    result = review_knowledge(
        knowledge_path,
        tmp_path / "review.json",
        tmp_path / "diagnostic.json",
        SequenceClient(responses),
        embedder=FakeEmbedder({}),
    )

    assert [review.knowledge_id for review in result.reviews] == [
        "k_0001",
        "k_0002",
        "k_0003",
    ]


def test_content_quality_forces_meta_content_to_discard() -> None:
    item = KnowledgeItem(
        "k_0001", 0, "fact", "Thanks for watching and subscribe.", "",
        3, [1], "Thanks.",
    )
    original = KnowledgeReview("k_0001", "keep", 0.7, [], "none", "Useful.")

    result = enforce_content_quality(item, original)

    assert result.status == "discard"
    assert result.recommended_action == "discard"


def test_content_quality_forces_compound_statement_to_split() -> None:
    item = KnowledgeItem(
        "k_0001", 0, "fact",
        "Alpha is first and beta is second, but gamma is third and delta is fourth.",
        "", 3, [1], "Evidence.",
    )
    original = KnowledgeReview("k_0001", "keep", 0.7, [], "none", "Useful.")

    result = enforce_content_quality(item, original)

    assert result.status == "split"
    assert result.issues == ["contains_two_facts"]


def test_atomicity_heuristic_preserves_comparison_as_one_learning_unit() -> None:
    item = KnowledgeItem(
        "k_0001", 0, "comparison",
        "Yoda is subjective, but soda is less subjective and rahashi is between them.",
        "", 4, [1], "Evidence.",
    )
    original = KnowledgeReview("k_0001", "keep", 0.9, [], "none", "Useful.")

    result = enforce_content_quality(item, original)

    assert result.status == "keep"


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
