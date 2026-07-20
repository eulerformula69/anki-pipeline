from fastapi.testclient import TestClient

from src.gui.app import create_app


def test_index_has_sources_stages_and_no_export(tmp_path):
    response = TestClient(create_app(root=tmp_path)).get("/")
    assert response.status_code == 200
    assert "YouTube URL" in response.text
    assert "Выбрать видеофайл" in response.text
    assert 'value="transcription"' in response.text
    assert 'name="knowledge_language" value="en" checked' in response.text
    assert 'name="knowledge_language" value="ru"' in response.text
    assert "Запустить конвейер" in response.text
    assert "build_anki" not in response.text


def test_review_ui_uses_candidate_thumbnails_and_non_sticky_toolbar():
    from pathlib import Path

    package = Path(__file__).parents[1] / "src" / "gui"
    template = (package / "templates" / "media_controls.html").read_text(encoding="utf-8")
    css = (package / "static" / "app.css").read_text(encoding="utf-8")

    assert "image-candidates/{{ loop.index0 }}" in template
    assert 'class="candidate"' in template
    assert template.count('hx-swap="outerHTML"') == 4
    assert "Показать варианты кадров" in template
    assert "{% if not card.alternatives %}" in template
    assert ".review-toolbar{position:sticky" not in css
    export = (package / "templates" / "export_panel.html").read_text(encoding="utf-8")
    assert 'name="deck_name"' in export
    assert 'name="global_tags"' in export
    assert "confirm(" not in export
    assert 'hx-target="#export-panel"' in export
