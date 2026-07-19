from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class AudioExtractionError(RuntimeError):
    pass


def ensure_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise AudioExtractionError(
            "FFmpeg не найден в PATH. Установите FFmpeg и перезапустите терминал."
        )
    return executable


def extract_audio(video: Path, audio: Path, *, force: bool = False) -> None:
    if audio.exists() and not force:
        return

    ffmpeg = ensure_ffmpeg()
    audio.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if force else "-n",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or "неизвестная ошибка FFmpeg"
        raise AudioExtractionError(f"Не удалось извлечь аудио: {details}") from exc

