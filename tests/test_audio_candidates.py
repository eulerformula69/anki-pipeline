import io
import wave

import pytest
from fastapi.testclient import TestClient

from src import audio_candidates, audio_sources
from src.audio_sources import AudioDownload, AudioLookupError, validate_audio
from src.gui.app import create_app


def test_audio_validation_rejects_short_playable_audio_and_fake_mp3(speech_mp3):
    validate_audio(speech_mp3)
    short = io.BytesIO()
    with wave.open(short, "wb") as audio:
        audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 1600)
    for data in (short.getvalue(), b"ID3fake", b"<html>error</html>", b""):
        with pytest.raises(AudioLookupError):
            validate_audio(data)


def test_bad_dictionary_response_falls_back_to_commons(monkeypatch, speech_mp3):
    expected = AudioDownload(speech_mp3, ".mp3", "", "CC0", "speaker")
    monkeypatch.setattr(audio_sources, "_request_bytes", lambda url: b"ID3invalid")
    monkeypatch.setattr(audio_sources, "find_commons_audio", lambda value: expected)
    assert audio_sources.find_native_audio("広告", "こうこく") == expected


def test_candidates_are_previewed_then_selected_without_replacing_original(tmp_path, monkeypatch, speech_mp3):
    native = AudioDownload(speech_mp3, ".mp3", "https://example.com", "", "speaker", "JapanesePod101")
    synthetic = AudioDownload(speech_mp3, ".mp3", "", "", "Microsoft", "Синтез речи · Nanami")
    monkeypatch.setattr(audio_candidates, "find_jpod101_audio", lambda *args: native)
    monkeypatch.setattr(audio_candidates, "find_commons_audio", lambda *args: None)
    monkeypatch.setattr(audio_candidates, "generate_audio", lambda *args: synthetic)
    client = TestClient(create_app(root=tmp_path))
    csrf = client.get("/").cookies["anki_csrf"]
    project_id = client.app.state.projects.create("Audio", "1 | recall | Перевод? | 広告[こうこく]")["id"]
    store = client.app.state.projects
    store.attach_audio(project_id, "card_0001", native, source="Original")
    url = f"/projects/{project_id}/cards/card_0001"
    data = {"csrf": csrf, "audio_text": "広告", "audio_reading": "こうこく", "included": "true"}
    assert client.post(url, data={**data, "audio_action": "candidates", "csrf": "bad"}).status_code == 403
    response = client.post(url, data={**data, "audio_action": "candidates"})
    assert response.status_code == 200
    assert "Запись не найдена." in response.text
    assert store.load(project_id)["cards"][0]["audio"]["source"] == "Original"
    card = store.load(project_id)["cards"][0]
    items = audio_candidates.read_candidates(store.root / project_id, card)
    token = items[2]["token"]
    preview = client.get(f"{url}/audio/candidates/{token}")
    assert preview.content == speech_mp3
    assert preview.headers["content-type"] == "audio/mpeg"
    assert client.get(f"{url}/audio/candidates/{'0' * 32}").status_code == 404
    assert client.post(url, data={**data, "audio_action": f"select:{token}"}).status_code == 200
    assert store.load(project_id)["cards"][0]["audio"]["source"] == synthetic.source_name
    assert client.get(f"{url}/audio").headers["cache-control"] == "no-store"
    changed = client.post(url, data={**data, "audio_text": "猫", "audio_action": f"select:{token}"})
    assert changed.status_code == 422
    assert store.load(project_id)["cards"][0]["audio"]["text"] == "広告"
    assert client.get(f"{url}/audio/candidates/{token}").status_code == 404
    assert client.post(url, data=data, files={"audio_file": ("bad.mp3", b"ID3fake")}).status_code == 422


def test_failed_source_does_not_hide_nanami_and_repeat_search_removes_old_files(tmp_path, monkeypatch, speech_mp3):
    def broken(*args):
        raise AudioLookupError("Повреждённый ответ.")

    monkeypatch.setattr(audio_candidates, "find_jpod101_audio", broken)
    monkeypatch.setattr(audio_candidates, "find_commons_audio", lambda *args: None)
    monkeypatch.setattr(audio_candidates, "generate_audio", lambda *args: AudioDownload(
        speech_mp3, ".mp3", "", "", "Nanami", "Nanami"
    ))
    audio_candidates.find_candidates(tmp_path, "card_0001", "広告", "こうこく")
    card = {"id": "card_0001", "audio_text": "広告", "audio_reading": "こうこく"}
    items = audio_candidates.read_candidates(tmp_path, card)
    assert items[0]["error"] == "Повреждённый ответ."
    old = items[2]["token"]
    audio_candidates.find_candidates(tmp_path, "card_0001", "広告", "こうこく")
    assert not (tmp_path / "audio-candidates" / old).exists()
    with pytest.raises(AudioLookupError):
        audio_candidates.selected_candidate(tmp_path, "card_0001", old, "広告", "こうこく")
