import re
import sqlite3
import sys
import types
import zipfile
from pathlib import Path

import pytest

from src.anki_export import (
    AnkiExportError,
    _anki_html_text,
    _media_filename,
    build_anki_deck,
    stable_guid,
)


class FakeModel:
    def __init__(self, model_id, name, **kwargs):
        self.model_id, self.name, self.kwargs = model_id, name, kwargs


class FakeNote:
    def __init__(self, model, fields, tags):
        self.model, self.fields, self.tags = model, fields, tags


class FakeDeck:
    def __init__(self, deck_id, name):
        self.deck_id, self.name, self.notes = deck_id, name, []

    def add_note(self, note):
        self.notes.append(note)


class FakePackage:
    last = None

    def __init__(self, deck):
        self.deck, self.media_files = deck, []
        FakePackage.last = self

    def write_to_file(self, path):
        database_path = Path(path).with_suffix(".anki2")
        database = sqlite3.connect(database_path)
        try:
            database.execute("CREATE TABLE cards (id INTEGER PRIMARY KEY)")
            database.executemany("INSERT INTO cards DEFAULT VALUES", (() for _ in self.deck.notes))
            database.commit()
        finally:
            database.close()
        with zipfile.ZipFile(path, "w") as package:
            package.write(database_path, "collection.anki2")
        database_path.unlink()


@pytest.fixture
def fake_genanki(monkeypatch):
    module = types.SimpleNamespace(Model=FakeModel, Note=FakeNote, Deck=FakeDeck, Package=FakePackage)
    monkeypatch.setitem(sys.modules, "genanki", module)


def project() -> dict:
    return {
        "id": "test-1234",
        "name": "Test",
        "created_at": "2026-10-11T12:00:00Z",
        "cards": [{
            "id": "card_0001",
            "type": "recall",
            "front": "A & B",
            "back": r"\(x<2\)",
            "extra": "",
            "tags": ["LLM-source"],
            "included": True,
        }],
    }


def test_export_builds_checked_deck_and_preserves_math(tmp_path, fake_genanki) -> None:
    destination = tmp_path / "deck.apkg"

    build_anki_deck(project(), destination, deck_name="Test", global_tags=["lesson 1"])

    assert destination.is_file()
    note = FakePackage.last.deck.notes[0]
    assert note.fields[:2] == ["A &amp; B", r"\(x&lt;2\)"]
    assert note.tags == ["LLM-source", "lesson_1"]
    assert note.guid == stable_guid("test-1234", "card_0001")
    assert "furigana-tap-toggle" in note.model.kwargs["templates"][0]["qfmt"]
    assert note.fields[-1] == ""
    assert "{{Audio}}" in note.model.kwargs["templates"][0]["afmt"]


def test_export_packages_attached_audio(tmp_path, fake_genanki) -> None:
    value = project()
    media = tmp_path / "media" / "card_0001.mp3"
    media.parent.mkdir()
    media.write_bytes(b"ID3audio")
    value["cards"][0]["audio"] = {
        "path": "media/card_0001.mp3",
        "text": "",
        "reading": "",
        "source": "manual",
        "source_url": "",
        "license": "personal",
        "attribution": "",
    }

    build_anki_deck(value, tmp_path / "deck.apkg", deck_name="Test", global_tags=[])

    audio_field = FakePackage.last.deck.notes[0].fields[-1]
    assert re.fullmatch(r"\[sound:anki_audio_20261011_[0-9a-f]{24}\.mp3\]", audio_field)
    assert Path(FakePackage.last.media_files[0]).name == audio_field.removeprefix(
        "[sound:"
    ).removesuffix("]")


def test_media_names_differ_between_projects_and_cards(tmp_path) -> None:
    source = tmp_path / "card_0012.mp3"
    source.write_bytes(b"ID3same-audio")
    first = project()
    second = project()
    second["id"] = "other-project"

    first_name = _media_filename(first, first["cards"][0], source)
    second_name = _media_filename(second, second["cards"][0], source)

    assert first_name != second_name
    assert first_name.startswith("anki_audio_20261011_")


def test_export_rejects_empty_selection(tmp_path) -> None:
    value = project()
    value["cards"][0]["included"] = False

    with pytest.raises(AnkiExportError, match="хотя бы одну"):
        build_anki_deck(value, tmp_path / "deck.apkg", deck_name="Test", global_tags=[])


def test_text_escaping_keeps_line_breaks() -> None:
    assert _anki_html_text("one\ntwo < three") == "one<br>two &lt; three"


def test_real_genanki_package_contains_a_card(tmp_path) -> None:
    pytest.importorskip("genanki")
    destination = tmp_path / "deck.apkg"

    build_anki_deck(project(), destination, deck_name="Test", global_tags=[])

    assert destination.is_file()
    with zipfile.ZipFile(destination) as package:
        database_name = next(
            name for name in ("collection.anki21b", "collection.anki2")
            if name in package.namelist()
        )
        database_path = tmp_path / database_name
        database_path.write_bytes(package.read(database_name))
    database = sqlite3.connect(database_path)
    try:
        assert database.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 1
    finally:
        database.close()
