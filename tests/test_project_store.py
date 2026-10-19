from src.audio_sources import AudioDownload
from src.project_store import ProjectStore
import pytest
from src.project_store import ProjectError
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event


def test_markdown_table_creates_and_appends_cards(tmp_path):
    store = ProjectStore(tmp_path)
    header = "| id | type | question | answer |\n| -- | -- | -- | -- |\n"
    project = store.create("Japanese", header + "| 1 | recall | Author? | 著者[ちょしゃ] |")
    store.append(project["id"], header + "| 2 | application | Translate. | このレポートを 読[よ]みます。 |")

    cards = store.load(project["id"])["cards"]

    assert [card["source_id"] for card in cards] == ["1", "2"]
    assert cards[0]["back"] == "著者[ちょしゃ]"
    assert cards[0]["audio_text"] == "著者"
    assert cards[1]["back"] == "このレポートを 読[よ]みます。"


def test_delete_preserves_other_cards_and_never_reuses_ids(tmp_path):
    store = ProjectStore(tmp_path)
    project_id = store.create("Delete", "1 | recall | Q | A\n2 | recall | Q2 | A2")["id"]
    original = store.load(project_id)["cards"][1]
    store.delete_card(project_id, "card_0001")
    assert store.load(project_id)["cards"] == [original]
    store.append(project_id, "3 | recall | Q3 | A3")
    assert store.load(project_id)["cards"][-1]["id"] == "card_0003"
    store.delete_card(project_id, "card_0003")
    store.delete_card(project_id, "card_0002")
    assert store.load(project_id)["cards"] == []
    assert store.add_card(project_id) == "card_0004"
    with pytest.raises(ProjectError, match="Карточка не найдена"):
        store.delete_card(project_id, "card_0001")


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


def test_project_stores_audio_file_and_metadata(tmp_path) -> None:
    store = ProjectStore(tmp_path / "output")
    project = store.create("Japanese", "1 | recall | Значение? | 女[おんな]")

    store.attach_audio(
        project["id"],
        "card_0001",
        AudioDownload(b"OggSdata", ".ogg", "https://example.test/file", "CC0", "Speaker"),
        source="Wikimedia Commons",
    )

    saved = store.load(project["id"])
    assert saved["cards"][0]["audio_text"] == "女"
    assert saved["cards"][0]["audio_reading"] == "おんな"
    assert saved["cards"][0]["audio"]["license"] == "CC0"
    assert (tmp_path / "output" / project["id"] / "media" / "card_0001.ogg").read_bytes() == b"OggSdata"


@pytest.mark.parametrize("operation", ["update", "append", "add", "delete", "move", "audio"])
def test_concurrent_changes_preserve_both_operations(tmp_path, monkeypatch, operation):
    store = ProjectStore(tmp_path)
    project_id = store.create("Concurrent", "1 | recall | Q1 | A1\n2 | recall | Q2 | A2")["id"]
    first_read = Event()
    release = Event()
    original_load = store.load

    def delayed_load(project_id):
        project = original_load(project_id)
        if not first_read.is_set():
            first_read.set()
            assert release.wait(5)
        return project

    monkeypatch.setattr(store, "load", delayed_load)
    updates = {"front": "Edited", "back": "A", "extra": "", "tags": [], "included": True}
    actions = {
        "update": lambda: store.update_card(project_id, "card_0002", updates),
        "append": lambda: store.append(project_id, "3 | recall | Q3 | A3"),
        "add": lambda: store.add_card(project_id),
        "delete": lambda: store.delete_card(project_id, "card_0002"),
        "move": lambda: store.move_card(project_id, "card_0002", "up"),
        "audio": lambda: store.attach_audio(project_id, "card_0002",
            AudioDownload(b"audio", ".mp3", "", "", ""), source="Test"),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(store.update_card, project_id, "card_0001", updates)
        try:
            assert first_read.wait(5)
            second = pool.submit(actions[operation])
            try:
                second.result(timeout=0.2)
            except TimeoutError:
                pass
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)

    cards = original_load(project_id)["cards"]
    by_id = {card["id"]: card for card in cards}
    assert by_id["card_0001"]["front"] == "Edited"
    if operation == "update":
        assert by_id["card_0002"]["front"] == "Edited"
    elif operation in {"append", "add"}:
        assert len(cards) == 3
        assert cards[-1]["id"] == "card_0003"
    elif operation == "delete":
        assert len(cards) == 1
    elif operation == "move":
        assert cards[0]["id"] == "card_0002"
    else:
        assert by_id["card_0002"]["audio"]["source"] == "Test"


@pytest.mark.parametrize("field", ["audio_text", "audio_reading"])
def test_outdated_audio_does_not_replace_existing_recording(tmp_path, field):
    store = ProjectStore(tmp_path)
    project_id = store.create("Audio", "1 | recall | Q | 猫[ねこ]")["id"]
    store.attach_audio(project_id, "card_0001", AudioDownload(b"original", ".mp3", "", "", ""), source="Original")
    card = store.load(project_id)["cards"][0]
    expected = (card["audio_text"], card["audio_reading"])
    store.update_card(project_id, "card_0001", {**card, field: "changed"})

    with pytest.raises(ProjectError, match="Добавьте аудио повторно"):
        store.attach_audio(project_id, "card_0001", AudioDownload(b"outdated", ".ogg", "", "", ""),
                           source="Generated", expected_audio=expected)

    assert store.audio_path(project_id, "card_0001").read_bytes() == b"original"
    assert store.load(project_id)["cards"][0]["audio"] == card["audio"]
