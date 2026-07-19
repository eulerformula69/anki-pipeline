import json

from src.models import Transcript, TranscriptSegment
from src.output import save_transcript_json


def test_save_transcript_json_as_utf8(tmp_path) -> None:
    transcript = Transcript(
        source="видео.mp4",
        language="ru",
        duration_seconds=5.2,
        model="small",
        segments=[TranscriptSegment(id=0, start=0.0, end=5.2, text="Привет")],
    )
    destination = tmp_path / "transcript.json"

    save_transcript_json(transcript, destination)

    assert "Привет" in destination.read_text(encoding="utf-8")
    assert json.loads(destination.read_text(encoding="utf-8"))["segments"][0]["id"] == 0

