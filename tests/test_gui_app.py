from pathlib import Path

from fastapi.testclient import TestClient

from src.gui.app import create_app


def test_index_starts_with_transcription_and_hides_legacy_stages(tmp_path):
    response = TestClient(create_app(root=tmp_path)).get("/")

    assert response.status_code == 200
    assert "Ссылка YouTube" in response.text
    assert "Видеофайл" in response.text
    assert 'type="hidden" name="stages" value="transcription"' in response.text
    assert "Создать транскрипцию" in response.text
    assert "NotebookLM" in response.text
    assert "LLM provider" not in response.text
    assert "Кандидаты карточек" not in response.text
    assert "build_anki" not in response.text


def test_job_ui_guides_notebooklm_handoff():
    template = (Path(__file__).parents[1] / "src" / "gui" / "templates" / "job.html").read_text(encoding="utf-8")

    assert "Скачать транскрипцию" in template
    assert "Скачать промпт" not in template
    assert "Скопировать промпт" in template
    assert "navigator.clipboard.writeText" in template
    assert 'id="notebooklm-prompt"' in template
    assert 'name="cards_file"' in template
    assert 'name="frames_per_card" value="5"' in template


def test_recent_project_is_openable(tmp_path):
    output = tmp_path / "output" / "lesson"
    output.mkdir(parents=True)
    (output / "transcript.json").write_text(
        '{"source":"lesson.mp4","segments":[]}', encoding="utf-8"
    )
    client = TestClient(create_app(root=tmp_path))

    index = client.get("/")
    assert 'class="project-row" href="/projects/lesson"' in index.text

    opened = client.get("/projects/lesson", follow_redirects=False)
    assert opened.status_code == 303
    assert opened.headers["location"].startswith("/jobs/")

    page = client.get(opened.headers["location"])
    assert "Скопировать промпт" in page.text
    assert "You are selecting and designing high-quality Anki cards" in page.text


def test_prompt_endpoint_is_plain_text_not_a_download(tmp_path):
    output = tmp_path / "output" / "lesson"
    output.mkdir(parents=True)
    (output / "transcript.json").write_text(
        '{"source":"lesson.mp4","segments":[]}', encoding="utf-8"
    )
    client = TestClient(create_app(root=tmp_path))
    opened = client.get("/projects/lesson", follow_redirects=False)
    job_url = opened.headers["location"]

    response = client.get(f"{job_url}/notebooklm-prompt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "content-disposition" not in response.headers


def test_review_ui_uses_candidate_thumbnails_and_non_sticky_toolbar():
    package = Path(__file__).parents[1] / "src" / "gui"
    template = (package / "templates" / "media_controls.html").read_text(encoding="utf-8")
    css = (package / "static" / "app.css").read_text(encoding="utf-8")

    assert "image-candidates/{{ loop.index0 }}" in template
    assert 'class="candidate"' in template
    assert template.count('hx-swap="outerHTML"') == 4
    assert "Показать варианты кадров" in template
    assert "{% if not card.alternatives %}" in template
    assert ".review-toolbar{position:sticky" not in css
    review = (package / "templates" / "review.html").read_text(encoding="utf-8")
    assert "card-type-badge" in review
    assert "card.meta.card_type|upper" in review
    export = (package / "templates" / "export_panel.html").read_text(encoding="utf-8")
    assert 'name="deck_name"' in export
    assert 'name="global_tags"' in export
    assert "confirm(" not in export
    assert 'hx-target="#export-panel"' in export


def test_htmx_errors_show_the_server_message_instead_of_claiming_image_failure():
    script = (Path(__file__).parents[1] / "src" / "gui" / "static" / "htmx.min.js").read_text(encoding="utf-8")

    assert "errorMessage(html)" in script
    assert "Не удалось выполнить действие" in script
    assert "Не удалось изменить изображение" not in script
