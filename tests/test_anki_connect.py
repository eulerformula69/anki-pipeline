import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from src.anki_connect import AnkiConnect, AnkiConnectError, load_settings, validate_url
from src.anki_export import FIELD_NAMES, MODEL_NAME
from src.gui.app import create_app
from src.project_store import ProjectStore


class FakeAnki(AnkiConnect):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.notes = {}
        self.model_exists = False
        self.fail_add = False

    def call(self, action, **params):
        self.calls.append((action, params))
        if action == "deckNames":
            return ["日本語", "Default"]
        if action == "modelNames":
            return [MODEL_NAME] if self.model_exists else []
        if action == "createModel":
            self.model_exists = True
        if action == "modelFieldNames":
            return list(FIELD_NAMES)
        if action == "findNotes":
            return [note_id for note_id, note in self.notes.items()
                    if note["tags"][-1] in params["query"]]
        if action == "storeMediaFile":
            return params["filename"]
        if action == "addNote":
            if self.fail_add:
                raise AnkiConnectError("Test failure")
            note_id = len(self.notes) + 1
            self.notes[note_id] = params["note"]
            return note_id
        if action == "notesInfo":
            return [{"modelName": MODEL_NAME, "cards": [100]}]


def test_import_repeat_updates_and_moves_with_audio(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("Japanese", "1 | recall | 寿司[すし] | sushi")
    directory = tmp_path / project["id"]
    (directory / "audio.mp3").write_bytes(b"audio")
    project["cards"][0]["audio"] = {"path": "audio.mp3"}
    anki = FakeAnki()

    assert anki.import_project(project, directory, "日本語", ["lesson"]) == (1, 0)
    note = anki.notes[1]
    assert note["deckName"] == "日本語"
    assert note["fields"]["Front"] == "寿司[すし]"
    assert note["fields"]["Audio"].startswith("[sound:anki_audio_")
    media = next(params for action, params in anki.calls if action == "storeMediaFile")
    assert base64.b64decode(media["data"]) == b"audio"
    model = next(params for action, params in anki.calls if action == "createModel")
    assert "furigana:Front" in model["cardTemplates"][0]["Front"]

    project["cards"][0]["back"] = "Updated"
    assert anki.import_project(project, directory, "Default", []) == (0, 1)
    assert len(anki.notes) == 1
    assert ("changeDeck", {"cards": [100], "deck": "Default"}) in anki.calls
    assert next(params for action, params in anki.calls if action == "updateNoteFields")["note"]["fields"]["Back"] == "Updated"


def test_validation_happens_before_mutations(tmp_path):
    project = ProjectStore(tmp_path).create("Test", "1 | recall | Q | A")
    anki = FakeAnki()
    with pytest.raises(AnkiConnectError, match="Колоды"):
        anki.import_project(project, tmp_path / project["id"], "Missing", [])
    project["cards"][0]["included"] = False
    with pytest.raises(AnkiConnectError, match="Включите"):
        anki.import_project(project, tmp_path / project["id"], "Default", [])
    assert all(action == "deckNames" for action, _ in anki.calls)


def test_partial_failure_can_be_retried(tmp_path):
    project = ProjectStore(tmp_path).create("Test", "1 | recall | Q | A\n2 | recall | Q2 | A2")
    anki = FakeAnki()
    original = anki.call

    def fail_second(action, **params):
        if action == "addNote" and len(anki.notes) == 1:
            raise AnkiConnectError("Connection lost")
        return original(action, **params)

    anki.call = fail_second
    with pytest.raises(AnkiConnectError, match="Добавлено: 1"):
        anki.import_project(project, tmp_path / project["id"], "Default", [])
    anki.call = original
    assert anki.import_project(project, tmp_path / project["id"], "Default", []) == (1, 1)


@pytest.mark.parametrize("url", ["file:///tmp", "http://host/path", "http://host?x=1", "http://host:abc", "http://u:p@host", "http://host:0"])
def test_invalid_addresses(url):
    with pytest.raises(AnkiConnectError):
        validate_url(url)


def test_protocol_errors_and_api_key(monkeypatch):
    def post(url, **kwargs):
        assert kwargs["json"]["key"] == "secret"
        assert kwargs["trust_env"] is False
        return httpx.Response(200, json={"error": "wrong key", "result": None},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(AnkiConnectError, match="wrong key"):
        AnkiConnect(key="secret").decks()


def test_connection_failure(monkeypatch):
    def post(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(AnkiConnectError, match="Нет связи"):
        AnkiConnect().decks()


@pytest.mark.parametrize("body", [[], {}, {"result": []}, {"error": None, "result": "wrong"}])
def test_malformed_response(monkeypatch, body):
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(
        200, json=body, request=httpx.Request("POST", url)))
    with pytest.raises(AnkiConnectError):
        AnkiConnect().decks()


def test_changed_model_stops_before_notes(tmp_path):
    project = ProjectStore(tmp_path).create("Test", "1 | recall | Q | A")
    anki = FakeAnki()
    anki.model_exists = True
    original = anki.call
    anki.call = lambda action, **params: ["Changed"] if action == "modelFieldNames" else original(action, **params)
    with pytest.raises(AnkiConnectError, match="Поля шаблона"):
        anki.import_project(project, tmp_path / project["id"], "Default", [])
    assert not anki.notes


def test_settings_persist_and_routes_check_csrf(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    client = TestClient(app)
    token = client.get("/").cookies["anki_csrf"]
    assert client.post("/anki-connect/settings", data={"csrf": "wrong", "url": "http://localhost:8765"}).status_code == 403
    monkeypatch.setattr(AnkiConnect, "decks", lambda self: ["Default"])
    response = client.post("/anki-connect/settings", data={"csrf": token, "url": "http://localhost:8765/", "key": "secret"})
    assert response.status_code == 200
    assert "secret" not in response.text
    assert load_settings(app.state.projects.root) == {"url": "http://localhost:8765", "key": "secret"}
    assert client.post("/anki-connect/decks", data={"csrf": token}).json()["decks"] == ["Default"]


def test_import_route_and_offline_settings(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    client = TestClient(app)
    token = client.get("/").cookies["anki_csrf"]
    project = app.state.projects.create("Test", "1 | recall | Q | A")
    anki = FakeAnki()
    monkeypatch.setattr(AnkiConnect, "call", lambda self, action, **params: anki.call(action, **params))
    url = f"/projects/{project['id']}/anki-connect"
    assert client.post(url, data={"csrf": "wrong", "deck_name": "Default"}).status_code == 403
    response = client.post(url, data={"csrf": token, "deck_name": "Default"})
    assert response.status_code == 200
    assert "Импорт в Anki завершён" in response.text
    assert client.post(url, data={"csrf": token, "deck_name": "Unknown"}).status_code == 422

    def offline(self):
        raise AnkiConnectError("Offline")

    monkeypatch.setattr(AnkiConnect, "decks", offline)
    response = client.post("/anki-connect/settings", data={"csrf": token, "url": "http://localhost:1234"})
    assert response.status_code == 422
    assert load_settings(app.state.projects.root)["url"] == "http://127.0.0.1:8765"
