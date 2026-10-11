import pytest

from src.card_input import CardInputError, parse_llm_response, to_project_cards


def test_parser_keeps_math_and_answer_pipes() -> None:
    cards = parse_llm_response(
        "ID | type | question | answer\n"
        r"source-1 | application | Find \(x\). | \(x=1\) | \(x=2\)"
    )

    assert len(cards) == 1
    assert cards[0].back == r"\(x=1\) | \(x=2\)"


def test_parser_rejects_duplicate_ids_and_unknown_types() -> None:
    with pytest.raises(CardInputError, match="повторяется"):
        parse_llm_response("1 | recall | Q | A\n1 | cloze | Q2 | A2")
    with pytest.raises(CardInputError, match="неизвестный"):
        parse_llm_response("1 | basic | Q | A")


def test_project_cards_get_stable_internal_ids() -> None:
    parsed = parse_llm_response("abc | recall | Q | A")

    cards = to_project_cards(parsed, start=7)

    assert cards[0]["id"] == "card_0007"
    assert cards[0]["source_id"] == "abc"
    assert cards[0]["tags"] == ["LLM-source", "recall"]
    assert cards[0]["audio_text"] == ""
    assert cards[0]["audio"] is None
