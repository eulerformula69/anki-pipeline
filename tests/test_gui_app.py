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
    assert "LLM-источник" in response.text
    assert "NotebookLM" not in response.text
    assert "LLM provider" not in response.text
    assert "Кандидаты карточек" not in response.text
    assert "build_anki" not in response.text
    assert "Только карточки" in response.text
    assert 'action="/card-projects"' in response.text


def test_job_ui_guides_notebooklm_handoff():
    template = (Path(__file__).parents[1] / "src" / "gui" / "templates" / "job.html").read_text(encoding="utf-8")

    assert "Скачать транскрипцию" in template
    assert "Скачать промпт" not in template
    assert "Скопировать промпт" in template
    assert "navigator.clipboard.writeText" in template
    assert 'id="notebooklm-prompt"' in template
    assert 'name="cards_file"' in template
    assert 'name="cards_text"' in template
    assert 'name="cards_file" type="file" accept=".txt,text/plain" required' not in template
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
    assert 'id="notebooklm-prompt"' in page.text
    assert "| type | question | answer" in page.text


def test_recent_project_with_cards_opens_review_even_when_restored_job_is_completed(tmp_path):
    output = tmp_path / "output" / "lesson"
    output.mkdir(parents=True)
    (output / "transcript.json").write_text(
        '{"source":"lesson.mp4","segments":[]}', encoding="utf-8"
    )
    (output / "card_candidates.json").write_text(
        '{"source":"lesson.mp4","model":"test","card_candidates":[]}', encoding="utf-8"
    )
    client = TestClient(create_app(root=tmp_path))

    opened = client.get("/projects/lesson", follow_redirects=False)

    assert opened.status_code == 303
    assert opened.headers["location"].endswith("/review")


def test_review_ui_can_append_notebooklm_cards():
    template = (Path(__file__).parents[1] / "src" / "gui" / "templates" / "review.html").read_text(encoding="utf-8")

    assert "Добавить карточки из LLM-источника" in template
    assert 'name="append" value="true"' in template
    assert "Добавить и подобрать кадры" in template
    assert "+ Карточка" in template
    assert 'name="direction" value="up"' in template
    assert 'name="direction" value="down"' in template


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
    assert "| type | question | answer" in response.text


def test_notebooklm_cards_can_be_imported_from_pasted_text(tmp_path):
    output = tmp_path / "output" / "lesson"
    output.mkdir(parents=True)
    (output / "transcript.json").write_text(
        '{"source":"lesson.mp4","segments":[{"id":1,"start":0,"end":3,"text":"今日は寒い。"}]}',
        encoding="utf-8",
    )
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]
    opened = client.get("/projects/lesson", follow_redirects=False)

    response = client.post(
        f'{opened.headers["location"]}/notebooklm-cards',
        data={
            "csrf": token,
            "cards_text": "1 | application | Complete: 今日[きょう]は ... 寒[さむ]い。 | より",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert (output / "card_candidates.json").is_file()


def test_notebooklm_import_requires_exactly_one_input(tmp_path):
    output = tmp_path / "output" / "lesson"
    output.mkdir(parents=True)
    (output / "transcript.json").write_text(
        '{"source":"lesson.mp4","segments":[{"id":1,"start":0,"end":3,"text":"test"}]}',
        encoding="utf-8",
    )
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]
    opened = client.get("/projects/lesson", follow_redirects=False)

    response = client.post(
        f'{opened.headers["location"]}/notebooklm-cards',
        data={"csrf": token, "cards_text": "1 | recall | Question? | Answer"},
        files={"cards_file": ("cards.txt", b"1 | recall | Other? | Other", "text/plain")},
    )

    assert response.status_code == 422
    assert "не оба сразу" in response.text


def test_card_only_project_opens_review_without_video_or_frames(tmp_path):
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]

    response = client.post(
        "/card-projects",
        data={
            "csrf": token,
            "project_name": "Learning science",
            "cards_text": "1 | recall | What is retrieval practice? | Recalling information from memory.",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].endswith("/review")
    review = client.get(response.headers["location"])
    assert review.status_code == 200
    assert "What is retrieval practice?" in review.text
    assert "Показать варианты кадров" not in review.text
    assert "Добавить карточки" in review.text


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
