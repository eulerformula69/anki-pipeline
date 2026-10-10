from fastapi.testclient import TestClient

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
