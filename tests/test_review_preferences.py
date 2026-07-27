import json

from src.review_preferences import (
    build_review_preference_prompt,
    load_review_preference_history,
)


def write_project(
    directory,
    decisions: list[bool],
    *,
    status: str = "approved",
) -> None:
    directory.mkdir()
    candidates = []
    reviewed = []
    for index, included in enumerate(decisions, 1):
        card_id = f"c_{index:04d}"
        candidates.append({
            "card_id": card_id,
            "front": f"Raw question {directory.name} {index}?",
            "back": f"Raw answer {index}.",
        })
        reviewed.append({
            "card_id": card_id,
            "included": included,
            "front": f"Edited question {directory.name} {index}?",
            "back": f"Edited answer {index}.",
        })
    (directory / "card_candidates.json").write_text(
        json.dumps({"card_candidates": candidates}),
        encoding="utf-8",
    )
    (directory / "approved_cards.json").write_text(
        json.dumps({"status": status, "cards": reviewed}),
        encoding="utf-8",
    )


def test_history_uses_only_completed_other_projects(tmp_path) -> None:
    write_project(tmp_path / "completed", [True, True, True, False, False, False])
    write_project(tmp_path / "draft", [True, False], status="draft")
    write_project(tmp_path / "current", [True, False])

    history = load_review_preference_history(
        tmp_path,
        exclude_directory=tmp_path / "current",
    )

    assert history.is_usable
    assert history.project_count == 1
    assert history.approved_count == 3
    assert history.rejected_count == 3
    assert {item.front for item in history.examples} == {
        f"Edited question completed {index}?" for index in range(1, 7)
    }


def test_history_limits_examples_but_preserves_full_counts(tmp_path) -> None:
    write_project(tmp_path / "completed", [True] * 8 + [False] * 9)

    history = load_review_preference_history(
        tmp_path,
        accepted_limit=2,
        rejected_limit=3,
    )

    assert history.approved_count == 8
    assert history.rejected_count == 9
    assert len(history.examples) == 5


def test_preference_prompt_is_conservative_and_contains_decisions(tmp_path) -> None:
    write_project(tmp_path / "completed", [True, True, True, False, False, False])
    history = load_review_preference_history(tmp_path)

    prompt = build_review_preference_prompt(history)

    assert '"decision": "keep"' in prompt
    assert '"decision": "discard"' in prompt
    assert "When uncertain, keep it" in prompt
    assert "not factual sources" in prompt


def test_too_little_history_does_not_change_prompt(tmp_path) -> None:
    write_project(tmp_path / "completed", [True, False])
    history = load_review_preference_history(tmp_path)

    assert not history.is_usable
    assert build_review_preference_prompt(history) == ""
