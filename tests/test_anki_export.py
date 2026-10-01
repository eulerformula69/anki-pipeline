import json
import sqlite3
import sys
import types
import zipfile
from pathlib import Path

import pytest

from src.anki_export import (
    AnkiExportError,
    FIELD_NAMES,
    build_anki_deck,
    load_card_candidates,
    load_card_media,
    load_verified_expressions,
    stable_deck_id,
    stable_guid,
    stage_media_image,
)


def write_inputs(root: Path, *, with_image: bool = True) -> tuple[Path, Path, Path]:
    cards = root / "card_candidates.json"
    expressions = root / "target_expressions.json"
    media = root / "card_media.json"
    cards.write_text(json.dumps({
        "source": "lesson.mkv", "model": "qwen3:8b", "card_candidates": [{
            "card_id": "c_0001", "knowledge_ids": ["k_0001"], "card_type": "basic",
            "front": "What is A & B?", "back": "A < B", "extra": "context",
            "difficulty": 2, "importance": 4, "tags": ["definition"],
            "source_transcript_segment_ids": [1],
            "expression_ids": ["e_verified", "e_transcribed", "e_review"],
        }],
    }, ensure_ascii=False), encoding="utf-8")
    expressions.write_text(json.dumps({"source": "lesson.mkv", "expressions": [
        {"id": "e_verified", "language": "ja", "surface": "明日", "reading": "あした",
         "romanization": "ashita", "meaning": "tomorrow", "knowledge_ids": ["k_0001"],
         "source_transcript_segment_ids": [1], "verification": "verified", "confidence": .9,
         "type": "term"},
        {"id": "e_review", "language": "ja", "surface": "偽物", "reading": "にせもの",
         "romanization": "nisemono", "meaning": "unverified", "knowledge_ids": ["k_0001"],
         "source_transcript_segment_ids": [1], "verification": "needs_review", "confidence": .5,
         "type": "term"},
        {"id": "e_transcribed", "language": "ja", "surface": "晴れ", "reading": "はれ",
         "romanization": "hare", "meaning": "sunny", "knowledge_ids": ["k_0001"],
         "source_transcript_segment_ids": [1], "verification": "transcribed", "confidence": .8,
         "type": "term"},
    ]}, ensure_ascii=False), encoding="utf-8")
    image = root / "media" / "c_0001.jpg"
    if with_image:
        from PIL import Image

        image.parent.mkdir()
        Image.new("RGB", (32, 18), "red").save(image, format="JPEG")
    media.write_text(json.dumps({"source": "lesson.mkv", "cards": [{
        "card_id": "c_0001", "start": 12.5,
        "image": {"path": "media/c_0001.jpg"} if with_image else None,
    }]}), encoding="utf-8")
    return cards, expressions, media


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
        database = Path(path).with_suffix(".anki2")
        connection = sqlite3.connect(database)
        try:
            connection.execute("CREATE TABLE cards (id INTEGER PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO cards DEFAULT VALUES",
                (() for _ in self.deck.notes),
            )
            connection.commit()
        finally:
            connection.close()
        with zipfile.ZipFile(path, "w") as package:
            package.write(database, "collection.anki2")
        database.unlink()


@pytest.fixture
def fake_genanki(monkeypatch):
    module = types.SimpleNamespace(
        Model=FakeModel, Note=FakeNote, Deck=FakeDeck, Package=FakePackage,
    )
    monkeypatch.setitem(sys.modules, "genanki", module)
    return module


def test_stable_identifiers_are_deterministic_and_card_specific() -> None:
    assert stable_guid("lesson.mkv", "c_0001") == stable_guid("lesson.mkv", "c_0001")
    assert stable_guid("lesson.mkv", "c_0001") != stable_guid("lesson.mkv", "c_0002")
    assert stable_deck_id("lesson.mkv") == stable_deck_id("lesson.mkv")
    assert 1 << 30 <= stable_deck_id("lesson.mkv") < 1 << 31


def test_exported_model_renders_furigana_in_japanese_text_fields(tmp_path, fake_genanki) -> None:
    cards, expressions, media = write_inputs(tmp_path, with_image=False)

    build_anki_deck(
        cards, expressions, media, tmp_path / "lesson.apkg", tmp_path / "manifest.json",
    )

    template = FakePackage.last.deck.notes[0].model.kwargs["templates"][0]
    assert "{{furigana:Front}}" in template["qfmt"]
    assert "{{furigana:Back}}" in template["afmt"]
    assert "{{furigana:Extra}}" in template["afmt"]
    assert "{{furigana:TargetText}}" in template["afmt"]
    assert "{{Reading}}" in template["afmt"]


def test_exported_model_has_optional_audio_on_answer(tmp_path, fake_genanki) -> None:
    cards, expressions, media = write_inputs(tmp_path, with_image=False)

    build_anki_deck(
        cards, expressions, media, tmp_path / "lesson.apkg", tmp_path / "manifest.json",
    )

    note = FakePackage.last.deck.notes[0]
    template = note.model.kwargs["templates"][0]
    assert FIELD_NAMES[-1] == "Audio"
    assert note.fields[-1] == ""
    assert "{{Audio}}" not in template["qfmt"]
    assert '{{#Audio}}<div class="audio">{{Audio}}</div>{{/Audio}}' in template["afmt"]


def test_exported_model_hides_furigana_until_hover_without_animation(tmp_path, fake_genanki) -> None:
    cards, expressions, media = write_inputs(tmp_path, with_image=False)

    build_anki_deck(
        cards, expressions, media, tmp_path / "lesson.apkg", tmp_path / "manifest.json",
    )

    css = FakePackage.last.deck.notes[0].model.kwargs["css"]
    assert "@media (hover: hover)" not in css
    assert "ruby rt" in css
    assert "visibility: hidden" in css
    assert "ruby:hover rt" in css
    assert "visibility: visible" in css
    assert "transition:" not in css


def test_exported_model_toggles_one_furigana_reading_on_touch_clients(tmp_path, fake_genanki) -> None:
    cards, expressions, media = write_inputs(tmp_path, with_image=False)

    build_anki_deck(
        cards, expressions, media, tmp_path / "lesson.apkg", tmp_path / "manifest.json",
    )

    model = FakePackage.last.deck.notes[0].model.kwargs
    template = model["templates"][0]
    for side in (template["qfmt"], template["afmt"]):
        assert "<!-- furigana-tap-toggle -->" in side
        assert 'document.addEventListener("click", window.furiganaTapHandler)' in side
        assert 'document.querySelectorAll("ruby.furigana-visible")' in side
        assert 'tappedRuby.classList.add("furigana-visible")' in side
        assert "mousemove" not in side

    css = model["css"]
    assert ".mobile ruby:hover rt" in css
    assert ".mobile ruby.furigana-visible rt" in css


def test_loaders_filter_unverified_and_reject_unsafe_media(tmp_path) -> None:
    cards, expressions, media = write_inputs(tmp_path)
    source, loaded = load_card_candidates(cards)
    assert source == "lesson.mkv" and loaded[0].card_id == "c_0001"
    assert list(load_verified_expressions(expressions)) == ["e_verified", "e_transcribed"]
    data = json.loads(media.read_text(encoding="utf-8"))
    data["cards"][0]["image"]["path"] = "../secret.jpg"
    media.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AnkiExportError, match="относительным"):
        load_card_media(media, tmp_path)


def test_build_deck_escapes_html_packages_media_and_writes_manifest(tmp_path, fake_genanki) -> None:
    cards, expressions, media = write_inputs(tmp_path)
    destination = tmp_path / "lesson.apkg"
    manifest = tmp_path / "anki_export_manifest.json"
    exported = build_anki_deck(cards, expressions, media, destination, manifest)
    package = FakePackage.last
    note = package.deck.notes[0]
    assert destination.is_file()
    assert len(note.fields) == len(FIELD_NAMES)
    assert note.fields[0] == "What is A &amp; B?"
    assert note.fields[1] == "A &lt; B"
    assert "明日" in note.fields[3] and "偽物" not in note.fields[3]
    assert note.fields[6].startswith('<img src="')
    assert Path(package.media_files[0]).name == exported[0].media
    assert note.guid == stable_guid("lesson.mkv", "c_0001")
    result = json.loads(manifest.read_text(encoding="utf-8"))
    assert result["notes"][0]["expression_ids"] == ["e_verified", "e_transcribed"]
    assert result["notes"][0]["start_seconds"] == 12.5


def test_export_preserves_line_breaks_in_editable_card_fields(tmp_path, fake_genanki) -> None:
    cards, expressions, media = write_inputs(tmp_path, with_image=False)
    data = json.loads(cards.read_text(encoding="utf-8"))
    card = data["card_candidates"][0]
    card["front"] = "First line\nSecond & line"
    card["back"] = "Answer one\r\nAnswer two"
    card["extra"] = "Note one\rNote two"
    cards.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    build_anki_deck(
        cards, expressions, media, tmp_path / "lesson.apkg", tmp_path / "manifest.json",
    )

    fields = FakePackage.last.deck.notes[0].fields
    assert fields[0] == "First line<br>Second &amp; line"
    assert fields[1] == "Answer one<br>Answer two"
    assert fields[2] == "Note one<br>Note two"


def test_real_package_contains_a_card_when_front_uses_furigana(tmp_path) -> None:
    pytest.importorskip("genanki")
    cards, expressions, media = write_inputs(tmp_path, with_image=False)
    destination = tmp_path / "lesson.apkg"

    build_anki_deck(cards, expressions, media, destination, tmp_path / "manifest.json")

    with zipfile.ZipFile(destination) as package:
        database_path = tmp_path / "collection.anki2"
        database_path.write_bytes(package.read("collection.anki2"))
    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 1
    finally:
        connection.close()


def test_export_adds_global_tags(tmp_path, fake_genanki) -> None:
    cards, expressions, media = write_inputs(tmp_path, with_image=False)

    build_anki_deck(
        cards, expressions, media, tmp_path / "lesson.apkg", tmp_path / "manifest.json",
        global_tags=["japanese", "lesson 24"],
    )

    assert FakePackage.last.deck.notes[0].tags == ["definition", "japanese", "lesson_24"]
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["global_tags"] == ["japanese", "lesson 24"]


def test_export_image_is_resized_without_distortion(tmp_path) -> None:
    from PIL import Image

    source = tmp_path / "large.png"
    destination = tmp_path / "small.jpg"
    Image.new("RGB", (1920, 1080), "red").save(source)

    stage_media_image(source, destination)

    with Image.open(destination) as result:
        assert result.size == (710, 399)
        assert result.format == "JPEG"
    assert destination.stat().st_size <= 10 * 1024


def test_export_image_never_exceeds_ten_kibibytes(tmp_path) -> None:
    import numpy as np
    from PIL import Image

    source = tmp_path / "noisy.png"
    destination = tmp_path / "compressed.jpg"
    pixels = np.random.default_rng(42).integers(0, 256, (900, 1600, 3), dtype=np.uint8)
    Image.fromarray(pixels, "RGB").save(source)

    stage_media_image(source, destination)

    assert destination.stat().st_size <= 10 * 1024
    with Image.open(destination) as result:
        assert result.width <= 710
        assert result.height <= 400


def test_missing_optional_files_builds_text_only_deck(tmp_path, fake_genanki) -> None:
    cards, _, _ = write_inputs(tmp_path)
    exported = build_anki_deck(
        cards, tmp_path / "missing-expressions.json", tmp_path / "missing-media.json",
        tmp_path / "lesson.apkg", tmp_path / "manifest.json",
    )
    assert exported[0].expression_ids == []
    assert exported[0].media is None
    assert FakePackage.last.media_files == []


def test_package_failure_preserves_existing_outputs(tmp_path, fake_genanki, monkeypatch) -> None:
    cards, expressions, media = write_inputs(tmp_path)
    destination = tmp_path / "lesson.apkg"
    manifest = tmp_path / "manifest.json"
    destination.write_bytes(b"existing")
    manifest.write_text('{"existing": true}', encoding="utf-8")
    monkeypatch.setattr(FakePackage, "write_to_file", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(AnkiExportError, match="boom"):
        build_anki_deck(cards, expressions, media, destination, manifest)
    assert destination.read_bytes() == b"existing"
    assert json.loads(manifest.read_text(encoding="utf-8")) == {"existing": True}
