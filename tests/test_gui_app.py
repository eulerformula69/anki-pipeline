from fastapi.testclient import TestClient

from src.gui.app import create_app


def test_index_has_sources_stages_and_no_export(tmp_path):
    response = TestClient(create_app(root=tmp_path)).get("/")
    assert response.status_code == 200
    assert "YouTube URL" in response.text
    assert "Выбрать видеофайл" in response.text
    assert 'value="transcription"' in response.text
    assert "Запустить конвейер" in response.text
    assert "build_anki" not in response.text
