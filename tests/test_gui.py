from fastapi.testclient import TestClient

from src.audio_sources import AudioDownload
from src.gui.app import create_app


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
    assert "Найти аудио" in page.text
    assert 'name="audio_text"' in page.text


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
        "src.gui.routes.find_commons_audio",
        lambda value: AudioDownload(b"OggS", ".ogg", "https://example.test", "CC0", "Speaker"),
    )

    found = client.post(f"{project_url}/audio/find", data={"csrf": token}, follow_redirects=False)

    assert "audio_found=1" in found.headers["location"]
    page = client.get(project_url)
    assert "Wikimedia Commons" in page.text
    assert "CC0" in page.text

    uploaded = client.post(
        f"{project_url}/cards/card_0001",
        data={
            "csrf": token,
            "front": "Значение?",
            "back": "寿司[すし]",
            "extra": "",
            "audio_text": "寿司",
            "tags": "japanese",
            "included": "true",
        },
        files={"audio_file": ("sushi.mp3", b"ID3audio", "audio/mpeg")},
        follow_redirects=False,
    )

    assert uploaded.status_code == 303
    project_id = project_url.rsplit("/", 1)[-1]
    card = client.app.state.projects.load(project_id)["cards"][0]
    assert card["audio"]["source"] == "Загружено вручную"
    assert card["audio"]["path"].endswith(".mp3")
