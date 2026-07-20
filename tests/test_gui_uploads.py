from fastapi.testclient import TestClient

from src.gui.app import create_app
from src.gui.schemas import JobState
from src.source_ingest import PreparedSource


class CapturingManager:
    def __init__(self):
        self.request = None

    def submit(self, request):
        self.request = request
        return JobState(job_id="job1", source=request.source, video_path=str(request.video_path), output_directory="output/video")


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


def test_youtube_prepared_string_path_is_converted_at_gui_boundary(tmp_path, monkeypatch):
    video = tmp_path / "input" / "downloads" / "video.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    manager = CapturingManager()
    monkeypatch.setattr("src.gui.routes.prepare_source", lambda value: PreparedSource(
        kind="youtube", video_path=str(video), source_url=str(value), source_id="abc",
        title="Lesson title", reused=False, prepared_at="2026-07-20T00:00:00Z",
    ))
    client = TestClient(create_app(root=tmp_path, manager=manager))
    token = client.get("/").cookies["anki_csrf"]
    response = client.post("/jobs", data={
        "csrf": token, "youtube_url": "https://youtu.be/abc", "stages": "transcription",
    }, follow_redirects=False)
    assert response.status_code == 303
    assert manager.request.video_path == video
    assert manager.request.source == "Lesson title"
