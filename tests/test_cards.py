import json

import pytest
import json
from pathlib import Path

from src.card_enrichment import candidates_are_duplicates
from src.cards import (
    CardGenerationError,
    CardProposal,
    deduplicate_candidates,
    generate_card_candidates,
    localize_existing_candidates,
    number_candidates,
    parse_card_response,
    proposal_to_candidate,
)
from src.models import CardCandidate, KnowledgeItem, KnowledgeReview, TargetExpression


def knowledge_items() -> list[KnowledgeItem]:
    return [
        KnowledgeItem("k_0001", 0, "definition", "Alpha is the first.", "", 4, [1], "Alpha."),
        KnowledgeItem("k_0002", 0, "fact", "Beta follows alpha.", "", 5, [2], "Beta."),
        KnowledgeItem("k_0003", 0, "example", "A weak example.", "", 2, [3], "Example."),
    ]


def review(knowledge_id: str, status: str) -> KnowledgeReview:
    return KnowledgeReview(knowledge_id, status, 0.9, [], "none", "Reason")


def card(ids, front="What defines alpha?", back="Alpha is the first element.", **overrides):
    value = {
        "knowledge_ids": ids,
        "card_type": "basic",
        "front": front,
        "back": back,
        "extra": "",
        "difficulty": 3,
    }
    value.update(overrides)
    return value


def test_keep_creates_card_and_discard_creates_none() -> None:
    items = knowledge_items()[:2]
    reviews = [review("k_0001", "keep"), review("k_0002", "discard")]
    data = {"card_candidates": [card(["k_0001"])]}

    result = parse_card_response(data, items, reviews)

    assert result[0].knowledge_ids == ["k_0001"]


def test_discard_card_is_rejected() -> None:
    items = knowledge_items()[:1]
    reviews = [review("k_0001", "discard")]

    with pytest.raises(CardGenerationError, match="discard|Discard"):
        parse_card_response({"card_candidates": [card(["k_0001"])]}, items, reviews)


def test_merge_multiple_knowledge_items() -> None:
    items = knowledge_items()[:2]
    reviews = [review("k_0001", "merge"), review("k_0002", "merge")]
    data = {
        "card_candidates": [
            card(["k_0001", "k_0002"], "How are alpha and beta related?", "Beta follows alpha.")
        ]
    }

    result = parse_card_response(data, items, reviews)
    candidate = proposal_to_candidate(result[0], {item.id: item for item in items})

    assert candidate.importance == 5
    assert candidate.source_transcript_segment_ids == [1, 2]


def test_split_creates_two_to_five_cards() -> None:
    items = knowledge_items()[:1]
    reviews = [review("k_0001", "split")]
    data = {
        "card_candidates": [
            card(["k_0001"], "What is alpha?", "Alpha is the first element."),
            card(["k_0001"], "Where does alpha appear?", "Alpha appears first."),
        ]
    }

    assert len(parse_card_response(data, items, reviews)) == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"card_type": "unknown"}, "card_type"),
        ({"front": ""}, "front пустой"),
        ({"back": ""}, "back пустой"),
        ({"front": "Q" * 121}, "front длиннее"),
        ({"back": "A" * 501}, "back длиннее"),
    ],
)
def test_invalid_card_fields(overrides, message) -> None:
    items = knowledge_items()[:1]
    reviews = [review("k_0001", "keep")]

    with pytest.raises(CardGenerationError, match=message):
        parse_card_response(
            {"card_candidates": [card(["k_0001"], **overrides)]}, items, reviews
        )


def test_duplicate_front_is_rejected() -> None:
    items = knowledge_items()[:2]
    reviews = [review("k_0001", "keep"), review("k_0002", "keep")]
    data = {
        "card_candidates": [
            card(["k_0001"]),
            card(["k_0002"], front="  WHAT DEFINES ALPHA?  "),
        ]
    }

    with pytest.raises(CardGenerationError, match="дублирующийся front"):
        parse_card_response(data, items, reviews)


def test_deduplication_and_stable_card_ids() -> None:
    first = CardCandidate("", ["k1"], "basic", "Question?", "Answer.", "", 2, 3, ["fact"], [1])
    second = CardCandidate("", ["k2"], "basic", " question? ", "Answer.", "", 4, 5, ["fact"], [2])

    result = deduplicate_candidates([first, second])

    assert len(result) == 1
    assert result[0].knowledge_ids == ["k1", "k2"]
    assert result[0].source_transcript_segment_ids == [1, 2]

    numbered = number_candidates([first, second])
    assert [candidate.card_id for candidate in numbered] == ["c_0001", "c_0002"]


def test_semantically_redundant_cards_are_deduplicated() -> None:
    first = CardCandidate(
        "", ["k1"], "basic", "What can そう be attached to?",
        "そう can only be attached to adjectival nouns.", "", 3, 4, ["fact"], [1],
    )
    second = CardCandidate(
        "", ["k2"], "basic", "What is そう attached to?",
        "The suffix そう is attached only to adjectival nouns.", "", 3, 4, ["fact"], [2],
    )

    assert candidates_are_duplicates(first, second)
    assert len(deduplicate_candidates([first, second])) == 1


def test_merge_source_with_discarded_duplicate_can_generate_one_card() -> None:
    items = [
        KnowledgeItem("k_0013", 0, "fact", "Keep this.", "", 4, [1], "Keep"),
        KnowledgeItem("k_0014", 0, "fact", "Duplicate.", "", 3, [2], "Duplicate"),
    ]
    reviews = [
        KnowledgeReview("k_0013", "keep", 0.9, [], "none", "Primary"),
        KnowledgeReview("k_0014", "discard", 0.9, ["redundant"], "discard", "Duplicate"),
    ]
    data = {"card_candidates": [{
        "knowledge_ids": ["k_0013"], "card_type": "basic",
        "front": "What should be kept?", "back": "Keep this.",
        "extra": "", "difficulty": 3,
    }]}

    result = parse_card_response(data, items, reviews)

    assert result[0].knowledge_ids == ["k_0013"]


def test_proposal_builds_automatic_fields() -> None:
    items = knowledge_items()[:2]
    proposal = CardProposal(
        ["k_0001", "k_0002"], "definition", "What is alpha?", "First.", "", 2
    )

    result = proposal_to_candidate(proposal, {item.id: item for item in items})

    assert result.importance == 5
    assert result.tags == ["definition", "fact"]
    assert result.source_transcript_segment_ids == [1, 2]


def test_expression_is_linked_and_native_form_is_added() -> None:
    items = knowledge_items()[:1]
    proposal = CardProposal(["k_0001"], "basic", "What does datte mean?", "A reason.", "", 2)
    expression = TargetExpression(
        "e_0001", "ja", "だって", "だって", "datte", "reason",
        ["k_0001"], [1], "needs_review", 0.9,
    )

    result = proposal_to_candidate(
        proposal, {items[0].id: items[0]}, {"k_0001": [expression]}
    )

    assert result.expression_ids == ["e_0001"]
    assert result.front == "What does だって mean?"
    assert "だって [だって] (datte)" in result.extra


def test_native_sentence_is_prefixed_when_romaji_is_misrecognized() -> None:
    items = knowledge_items()[:1]
    proposal = CardProposal(
        ["k_0001"], "basic", "What does ashto wa hereta mean?", "Tomorrow is sunny.", "", 2
    )
    expression = TargetExpression(
        "e_0002", "ja", "明日は晴れだって", "あしたははれだって",
        "ashita wa hare datte", "It is said tomorrow will be sunny.",
        ["k_0001"], [1], "verified", 0.9, "example_sentence",
    )

    result = proposal_to_candidate(
        proposal, {items[0].id: items[0]}, {"k_0001": [expression]}
    )

    assert result.front.startswith("明日は晴れだって — ")


def test_verified_term_is_replaced_globally() -> None:
    items = knowledge_items()[:1]
    proposal = CardProposal(
        ["k_0001"], "basic", "How does datte work?", "Datte marks a quotation.", "", 2
    )
    term = TargetExpression(
        "e_0001", "ja", "だって", "だって", "datte", "quotation",
        ["k_other"], [99], "verified", 0.9, "term",
    )

    result = proposal_to_candidate(
        proposal, {items[0].id: items[0]}, {}, [term]
    )

    assert result.front == "How does だって work?"
    assert result.back == "だって marks a quotation."
    assert result.expression_ids == []


def test_existing_card_is_relinked_to_expression_by_source_segment(tmp_path: Path) -> None:
    candidate = CardCandidate(
        "c_0001", ["k_0001"], "basic", "What does it mean?", "Meaning.", "",
        2, 4, ["basic"], [7], [],
    )
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(json.dumps({
        "source": "video", "model": "qwen3:8b",
        "card_candidates": [candidate.__dict__],
    }), encoding="utf-8")
    expression = TargetExpression(
        "e_0001", "ja", "らしい", "らしい", "rashii", "seems",
        ["k_0001"], [7], "transcribed", 0.9, "grammar_pattern",
    )
    expressions_path = tmp_path / "expressions.json"
    expressions_path.write_text(json.dumps({
        "source": "video", "model": "qwen3:8b", "target_language": "ja",
        "expressions": [expression.__dict__],
    }), encoding="utf-8")

    localize_existing_candidates(cards_path, expressions_path)

    saved = json.loads(cards_path.read_text(encoding="utf-8"))["card_candidates"][0]
    assert saved["front"].startswith("らしい — ")
    assert saved["expression_ids"] == ["e_0001"]


def test_plain_text_cloze_underscores_are_allowed() -> None:
    items = knowledge_items()[:1]
    reviews = [review("k_0001", "keep")]
    data = {
        "card_candidates": [
            card(
                ["k_0001"],
                front="Alpha appears ___ in the sequence.",
                back="Alpha appears first.",
                card_type="cloze",
            )
        ]
    }

    assert parse_card_response(data, items, reviews)[0].card_type == "cloze"


class SequenceClient:
    model = "qwen3:8b"

    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def generate_json(self, prompt: str):
        self.calls += 1
        return next(self.responses)


def write_card_sources(tmp_path):
    knowledge_path = tmp_path / "knowledge.json"
    review_path = tmp_path / "review.json"
    item = knowledge_items()[0]
    knowledge_path.write_text(
        json.dumps(
            {
                "source": "video.mkv",
                "model": "qwen3:8b",
                "knowledge_items": [item.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    review_path.write_text(
        json.dumps(
            {
                "source": "video.mkv",
                "model": "qwen3:8b",
                "reviews": [review("k_0001", "keep").__dict__],
            }
        ),
        encoding="utf-8",
    )
    return knowledge_path, review_path


def test_card_generation_retries_and_numbers_result(tmp_path) -> None:
    knowledge_path, review_path = write_card_sources(tmp_path)
    invalid = ({"card_candidates": [card(["k_0001"], card_type="wrong")]}, "bad")
    valid_data = {
        "card_candidates": [
            card(
                ["k_0001"],
                front="Как определить alpha?",
                back="Alpha является первым элементом.",
            )
        ]
    }
    client = SequenceClient([invalid, (valid_data, "valid")])
    destination = tmp_path / "cards.json"

    result = generate_card_candidates(
        knowledge_path,
        review_path,
        destination,
        tmp_path / "diagnostic.json",
        client,
    )

    assert client.calls == 2
    assert result.card_candidates[0].card_id == "c_0001"
    assert "Как определить" in destination.read_text(encoding="utf-8")


def test_invalid_cards_use_drafts_and_preserve_diagnostic(tmp_path) -> None:
    knowledge_path, review_path = write_card_sources(tmp_path)
    invalid = ({"card_candidates": [card(["k_0001"], card_type="wrong")]}, "bad")
    client = SequenceClient([invalid, invalid])
    destination = tmp_path / "cards.json"
    destination.write_text("existing", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic.json"

    result = generate_card_candidates(
        knowledge_path, review_path, destination, diagnostic, client
    )

    assert result.card_candidates
    assert "Automatic fallback" in result.card_candidates[0].extra
    assert len(json.loads(diagnostic.read_text(encoding="utf-8"))["ollama_responses"]) == 2
