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


@pytest.mark.parametrize("borders", [True, False])
def test_parser_accepts_markdown_tables(borders) -> None:
    rows = [
        "id | type | question | answer",
        ":-- | :--------------: | --- | --:",
        "1 | application | Translate: “I will check who wrote this report.” | このレポートを 誰[だれ]が 書[か]いたか 確認[かくにん]します。",
        r"2 | recall | What does \| mean? | A pipe: \|",
        "3 | discrimination | Choose か or かどうか. | かどうか",
    ]
    text = "\n".join(f"| {row} |" if borders else row for row in rows)

    cards = parse_llm_response(text)

    assert len(cards) == 3
    assert cards[0].source_id == "1"
    assert cards[0].back == "このレポートを 誰[だれ]が 書[か]いたか 確認[かくにん]します。"
    assert cards[1].front == "What does | mean?"
    assert cards[1].back == "A pipe: |"
    assert cards[2].card_type == "discrimination"


@pytest.mark.parametrize("row, message", [
    ("| 1 | recall | Q | |", "используйте"),
    ("| 1 | basic | Q | A |", "неизвестный"),
    ("| 1 | recall | Q | A |\n| 1 | recall | Q2 | A2 |", "повторяется"),
    ("| id | type | question | answer |\n| -- | -- | -- | -- |", "не содержит"),
])
def test_markdown_tables_keep_validation(row, message) -> None:
    with pytest.raises(CardInputError, match=message):
        parse_llm_response(row)


def test_project_cards_get_stable_internal_ids() -> None:
    parsed = parse_llm_response("abc | recall | Q | A")

    cards = to_project_cards(parsed, start=7)

    assert cards[0]["id"] == "card_0007"
    assert cards[0]["source_id"] == "abc"
    assert cards[0]["tags"] == ["LLM-source", "recall"]
    assert cards[0]["audio_text"] == ""
    assert cards[0]["audio_reading"] == ""
    assert cards[0]["audio"] is None
