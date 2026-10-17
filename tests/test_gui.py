import re
import zipfile
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.audio_sources import AudioDownload, AudioLookupError
from src.gui.app import create_app


@pytest.mark.parametrize("failure", [False, True])
def test_add_audio_for_one_card_saves_edits_and_leaves_other_cards(tmp_path, monkeypatch, failure):
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]
    project = client.app.state.projects.create(
        "Single audio", "1 | recall | Перевод? | 猫\n2 | recall | Перевод? | 犬"
    )
    project_id = project["id"]
    original = AudioDownload(b"original", ".mp3", "", "", "speaker")
    client.app.state.projects.attach_audio(project_id, "card_0002", original, source="Original")
    before = client.app.state.projects.load(project_id)["cards"][1]
    calls = []

    def lookup(text, reading):
        calls.append((text, reading))
        if failure:
            raise AudioLookupError("Сервис недоступен.")
        return AudioDownload(b"ID3single", ".mp3", "", "", "Nanami", "Nanami")

    monkeypatch.setattr("src.gui.routes.find_or_generate_audio", lookup)
    url = f"/projects/{project_id}/cards/card_0001"
    response = client.post(url, data={
        "csrf": token, "front": "Новый вопрос", "back": "絞り込む",
        "audio_text": "絞り込む", "audio_reading": "しぼりこむ",
        "included": "true", "action": "audio",
    }, follow_redirects=False)
    assert calls == [("絞り込む", "しぼりこむ")]
    cards = client.app.state.projects.load(project_id)["cards"]
    assert cards[0]["front"] == "Новый вопрос"
    assert cards[1] == before
    if failure:
        assert response.status_code == 422
        assert "Карточка сохранена" in response.text
        assert cards[0]["audio"] is None
    else:
        assert response.status_code == 303
        assert response.headers["location"].endswith("#card_0001")
        assert client.get(f"{url}/audio").content == b"ID3single"


def test_generated_audio_can_be_previewed_and_exported(tmp_path, monkeypatch):
    monkeypatch.setattr("src.audio_tts.find_native_audio", lambda *args: None)

    async def synthesize(text):
        assert text == "この情報が信頼できるかどうか調べています。"
        return b"ID3generated-speech"

    monkeypatch.setattr("src.audio_tts._synthesize", synthesize)
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]
    created = client.post("/projects", data={
        "csrf": token,
        "cards_text": "1 | recall | Перевод? | この情報が信頼できるかどうか調べています。",
    }, follow_redirects=False)
    url = created.headers["location"]
    result = client.post(f"{url}/audio/find", data={"csrf": token}, follow_redirects=False)
    assert "audio_found=1" in result.headers["location"]
    assert "Синтез речи · Nanami" in client.get(url).text
    preview = client.get(f"{url}/cards/card_0001/audio")
    assert preview.content == b"ID3generated-speech"
    assert preview.headers["content-type"] == "audio/mpeg"
    exported = client.post(f"{url}/export", data={"csrf": token, "deck_name": "Nanami"})
    assert exported.status_code == 200
    with zipfile.ZipFile(tmp_path / "output" / url.rsplit("/", 1)[-1] / "deck.apkg") as package:
        assert b"ID3generated-speech" in [package.read(name) for name in package.namelist()]


@pytest.mark.parametrize("name", [None, "", "   "])
def test_create_project_without_name(tmp_path, name) -> None:
    client = TestClient(create_app(root=tmp_path))
    index = client.get("/")
    name_input = re.search(r'<input name="project_name"[^>]*>', index.text).group()
    assert "required" not in name_input
    data = {"csrf": index.cookies["anki_csrf"], "cards_text": "1 | recall | Q | A"}
    if name is not None:
        data["project_name"] = name
    before = datetime.now().astimezone().date().isoformat()

    response = client.post("/projects", data=data, follow_redirects=False)

    after = datetime.now().astimezone().date().isoformat()
    assert response.status_code == 303
    project_id = response.headers["location"].rsplit("/", 1)[-1]
    project = client.app.state.projects.load(project_id)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{8}", project["name"])
    assert project["name"][:10] in {before, after}
    assert project_id == project["name"]
    assert project["name"] in client.get(response.headers["location"]).text


def test_full_card_flow(tmp_path, monkeypatch) -> None:
    client = TestClient(create_app(root=tmp_path))
    index = client.get("/")
    token = index.cookies["anki_csrf"]
    assert "Ответ LLM → Anki" in index.text
    assert "YouTube" not in index.text

    created = client.post(
        "/projects",
        data={
            "csrf": token,
            "project_name": "Test deck",
            "cards_text": "1 | recall | Question? | Answer",
        },
        follow_redirects=False,
    )

    assert created.status_code == 303
    page = client.get(created.headers["location"])
    assert "Question?" in page.text
    assert "Создать `.apkg`" in page.text
    assert "Добавить аудио" in page.text
    assert 'name="audio_text"' in page.text
    assert 'name="audio_reading"' in page.text
    assert "this.form.requestSubmit()" in page.text
    assert "Загружаем аудио…" in page.text
    assert "data-async-page" in page.text
    assert "Создаём `.apkg`…" in page.text
    script = client.get("/static/app.js?v=3")
    assert script.status_code == 200
    assert 'fetch(form.action' in script.text


def test_create_requires_exactly_one_card_input(tmp_path) -> None:
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]

    response = client.post(
        "/projects",
        data={"csrf": token, "project_name": "Test", "cards_text": "1 | recall | Q | A"},
        files={"cards_file": ("cards.txt", b"1 | recall | Other | Other", "text/plain")},
    )

    assert response.status_code == 422
    assert "один .txt-файл" in response.text


def test_prompt_is_opened_as_plain_text(tmp_path) -> None:
    response = TestClient(create_app(root=tmp_path)).get("/prompt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "| type | question | answer" in response.text


def test_audio_lookup_updates_project(tmp_path, monkeypatch) -> None:
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]
    created = client.post(
        "/projects",
        data={
            "csrf": token,
            "project_name": "Japanese",
            "cards_text": "1 | recall | Значение? | 寿司[すし]",
        },
        follow_redirects=False,
    )
    project_url = created.headers["location"]
    monkeypatch.setattr(
        "src.gui.routes.find_or_generate_audio",
        lambda value, reading: AudioDownload(
            b"OggS", ".ogg", "https://example.test", "CC0", "Speaker"
        ),
    )

    found = client.post(f"{project_url}/audio/find", data={"csrf": token}, follow_redirects=False)

    assert "audio_found=1" in found.headers["location"]
    page = client.get(project_url)
    assert "Wikimedia Commons" in page.text
    assert "CC0" in page.text
    assert "<audio" in page.text

    preview = client.get(f"{project_url}/cards/card_0001/audio")
    assert preview.status_code == 200
    assert preview.content == b"OggS"
    assert preview.headers["content-type"] == "audio/ogg"

    monkeypatch.setattr(
        "src.gui.routes.find_or_generate_audio",
        lambda *args: pytest.fail("Existing audio was regenerated"),
    )
    repeated = client.post(f"{project_url}/audio/find", data={"csrf": token}, follow_redirects=False)
    assert "audio_found=0" in repeated.headers["location"]

    uploaded = client.post(
        f"{project_url}/cards/card_0001",
        data={
            "csrf": token,
            "front": "Значение?",
            "back": "寿司[すし]",
            "extra": "",
            "audio_text": "寿司",
            "audio_reading": "すし",
            "tags": "japanese",
            "included": "true",
        },
        files={"audio_file": ("sushi.mp3", b"ID3audio", "audio/mpeg")},
        follow_redirects=False,
    )

    assert uploaded.status_code == 303
    assert "audio_uploaded=1" in uploaded.headers["location"]
    project_id = project_url.rsplit("/", 1)[-1]
    card = client.app.state.projects.load(project_id)["cards"][0]
    assert card["audio"]["source"] == "Загружено вручную"
    assert card["audio"]["path"].endswith(".mp3")


def test_audio_preview_returns_not_found_without_recording(tmp_path) -> None:
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]
    created = client.post(
        "/projects",
        data={
            "csrf": token,
            "project_name": "No audio",
            "cards_text": "1 | recall | Question? | Answer",
        },
        follow_redirects=False,
    )

    preview = client.get(f"{created.headers['location']}/cards/card_0001/audio")

    assert preview.status_code == 404
