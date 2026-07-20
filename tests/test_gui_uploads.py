from fastapi.testclient import TestClient

from src.gui.app import create_app


def test_empty_and_double_source_are_rejected(tmp_path):
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]
    empty = client.post("/jobs", data={"csrf": token})
    assert empty.status_code == 422
    both = client.post(
        "/jobs", data={"csrf": token, "youtube_url": "https://youtu.be/abc", "stages": "transcription"},
        files={"video_file": ("clip.mp4", b"video", "video/mp4")},
    )
    assert both.status_code == 422


def test_unknown_video_extension_is_rejected(tmp_path):
    client = TestClient(create_app(root=tmp_path))
    token = client.get("/").cookies["anki_csrf"]
    response = client.post(
        "/jobs", data={"csrf": token, "stages": "transcription"},
        files={"video_file": ("../bad.exe", b"x", "application/octet-stream")},
    )
    assert response.status_code == 422
