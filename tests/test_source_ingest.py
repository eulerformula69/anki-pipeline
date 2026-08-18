from pathlib import Path

import pytest

from src.source_ingest import (
    FILE_MARKER,
    SourceIngestError,
    normalize_youtube_url,
    prepare_local_source,
    prepare_youtube_source,
)


def test_only_youtube_http_urls_are_allowed() -> None:
    assert normalize_youtube_url("https://youtu.be/abc#fragment") == "https://youtu.be/abc"
    assert normalize_youtube_url("http://www.youtube.com/watch?v=abc").startswith("https://")
    with pytest.raises(SourceIngestError, match="Разрешены только"):
        normalize_youtube_url("https://example.com/video")
    with pytest.raises(SourceIngestError, match="Разрешены только"):
        normalize_youtube_url("file:///C:/secret.txt")


def test_local_source_is_preserved(tmp_path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"video")

    result = prepare_local_source(video)

    assert result.kind == "local"
    assert Path(result.video_path) == video.resolve()
    assert result.title == "lesson"


class FakeProcess:
    def __init__(self, lines, return_code=0):
        self.stdout = iter(lines)
        self.return_code = return_code

    def wait(self):
        return self.return_code


def test_youtube_download_writes_manifest_and_reuses_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.source_ingest.importlib.util.find_spec", lambda name: object())
    video = tmp_path / "YouTube" / "abc" / "video.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    video.with_suffix(".info.json").write_text(
        '{"id":"abc","title":"Stable download"}', encoding="utf-8",
    )
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        return FakeProcess(["[download] 50%\n", f"{FILE_MARKER}{video}\n"])

    first = prepare_youtube_source(
        "https://youtube.com/watch?v=abc", input_root=tmp_path,
        subtitle_languages=("en", "ja"), popen=fake_popen,
    )
    second = prepare_youtube_source(
        "https://youtube.com/watch?v=abc", input_root=tmp_path,
        popen=lambda *a, **k: pytest.fail("Повторная загрузка запрещена"),
    )

    assert first.reused is False and second.reused is True
    assert first.source_id == "abc" and first.title == "Stable download"
    assert (video.parent / "source_manifest.json").exists()
    assert (tmp_path / "source_index.json").exists()
    command = calls[0]
    assert "--no-playlist" in command
    assert "--no-remote-components" in command
    assert "en.*,ja.*" in command


def test_download_failure_includes_real_reason(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.source_ingest.importlib.util.find_spec", lambda name: object())

    with pytest.raises(SourceIngestError, match="HTTP Error 422"):
        prepare_youtube_source(
            "https://youtu.be/abc", input_root=tmp_path,
            popen=lambda *a, **k: FakeProcess(
                ["noise\n", "ERROR: HTTP Error 422: Unprocessable Entity\n"],
                return_code=1,
            ),
        )


def test_http_403_retries_with_embedded_client(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.source_ingest.importlib.util.find_spec", lambda name: object())
    video = tmp_path / "Youtube" / "abc" / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return FakeProcess(["ERROR: HTTP Error 403: Forbidden\n"], return_code=1)
        return FakeProcess([f"{FILE_MARKER}{video}\n"])

    result = prepare_youtube_source(
        "https://youtu.be/abc", input_root=tmp_path, popen=fake_popen,
    )

    assert result.video_path == str(video.resolve())
    assert len(calls) == 2
    assert "youtube:player_client=web_embedded" in calls[1]


def test_download_uses_retries_and_available_node(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.source_ingest.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("src.source_ingest.shutil.which", lambda name: "node.exe")
    video = tmp_path / "YouTube" / "abc" / "video.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        return FakeProcess([f"{FILE_MARKER}{video}\n"])

    prepare_youtube_source(
        "https://youtu.be/abc", input_root=tmp_path, popen=fake_popen,
    )

    command = calls[0]
    assert "--ignore-config" in command
    assert "--extractor-retries" in command
    assert command[command.index("--js-runtimes") + 1] == "node"


def test_downloaded_path_cannot_escape_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.source_ingest.importlib.util.find_spec", lambda name: object())
    outside = tmp_path.parent / "outside-video.mkv"
    outside.write_bytes(b"video")

    with pytest.raises(SourceIngestError, match="за пределами"):
        prepare_youtube_source(
            "https://youtu.be/abc", input_root=tmp_path,
            popen=lambda *a, **k: FakeProcess([f"{FILE_MARKER}{outside}\n"]),
        )
