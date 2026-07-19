from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .audio import AudioExtractionError, ensure_ffmpeg, extract_audio
from .output import save_transcript_json, save_transcript_text
from .paths import build_result_paths
from .transcription import TranscriptionError, resolve_device, transcribe_audio

LOGGER = logging.getLogger("anki_pipeline")


def validate_input(video: Path) -> None:
    if not video.exists():
        raise FileNotFoundError(f"Входной файл не найден: {video}")
    if not video.is_file():
        raise ValueError(f"Указанный путь не является файлом: {video}")


def run_pipeline(
    video: Path,
    *,
    model: str = "small",
    device: str = "auto",
    language: str | None = None,
    force: bool = False,
) -> Path:
    validate_input(video)
    ensure_ffmpeg()
    paths = build_result_paths(video)

    if paths.transcript_json.exists() and paths.transcript_txt.exists() and not force:
        LOGGER.info("Транскрипция уже существует: %s", paths.transcript_json)
        return paths.transcript_json

    if paths.audio.exists() and not force:
        LOGGER.info("Используется существующее аудио: %s", paths.audio)
    else:
        LOGGER.info("Извлечение mono-аудио 16 kHz: %s", paths.audio)
        extract_audio(video, paths.audio, force=force)

    selected_device, compute_type = resolve_device(device)
    LOGGER.info("Транскрибация: model=%s device=%s compute=%s", model, selected_device, compute_type)
    transcript = transcribe_audio(
        paths.audio,
        video,
        model_name=model,
        device=device,
        language=language,
    )
    save_transcript_json(transcript, paths.transcript_json)
    save_transcript_text(transcript, paths.transcript_txt)
    LOGGER.info("Готово: %s", paths.directory)
    return paths.transcript_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Локальная транскрибация видео")
    parser.add_argument("video", type=Path, help="Путь к видеофайлу")
    parser.add_argument("--model", default="small", help="Модель Whisper (default: small)")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="Устройство"
    )
    parser.add_argument("--language", help="Код языка, например ru; по умолчанию авто")
    parser.add_argument("--force", action="store_true", help="Перезаписать результаты")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        run_pipeline(
            args.video,
            model=args.model,
            device=args.device,
            language=args.language,
            force=args.force,
        )
    except (FileNotFoundError, ValueError, AudioExtractionError, TranscriptionError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

