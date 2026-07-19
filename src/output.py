from __future__ import annotations

import json
from pathlib import Path

from .models import Transcript


def seconds_to_timecode(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_transcript_json(transcript: Transcript, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_transcript_text(transcript: Transcript, destination: Path) -> None:
    lines = [
        f"[{seconds_to_timecode(segment.start)} - "
        f"{seconds_to_timecode(segment.end)}] {segment.text}"
        for segment in transcript.segments
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

