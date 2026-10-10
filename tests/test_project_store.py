from src.project_store import ProjectStore


def test_project_lifecycle_uses_one_json_file(tmp_path) -> None:
    store = ProjectStore(tmp_path / "output")
    project = store.create("Learning science", "1 | recall | Question? | Answer")

    store.update_card(
        project["id"],
        "card_0001",
        {"front": "Edited?", "back": "Yes", "extra": "Note", "tags": ["tag"], "included": True},
    )
    store.append(project["id"], "2 | application | Apply? | Done")
    manual_id = store.add_card(project["id"])
    store.move_card(project["id"], manual_id, "up")

    saved = store.load(project["id"])
    assert saved["cards"][0]["front"] == "Edited?"
    assert [card["id"] for card in saved["cards"]] == ["card_0001", "card_0003", "card_0002"]
    assert list((tmp_path / "output" / project["id"]).iterdir()) == [
        tmp_path / "output" / project["id"] / "project.json"
    ]


def test_projects_are_sorted_by_update_time(tmp_path) -> None:
    store = ProjectStore(tmp_path / "output")
    first = store.create("First", "1 | recall | Q | A")
    second = store.create("Second", "1 | recall | Q | A")
    store.update_card(
        first["id"], "card_0001",
        {"front": "Q", "back": "A", "extra": "", "tags": [], "included": True},
    )

    assert store.list()[0]["id"] == first["id"]
    assert {item["id"] for item in store.list()} == {first["id"], second["id"]}
