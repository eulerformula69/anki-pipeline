from __future__ import annotations

import asyncio

import edge_tts

from .audio_sources import (
    MAX_AUDIO_BYTES,
    AudioDownload,
    AudioLookupError,
    find_native_audio,
    strip_furigana,
    validate_audio,
)


VOICE = "ja-JP-NanamiNeural"


def find_or_generate_audio(value: str, reading: str) -> AudioDownload | None:
    try:
        recording = find_native_audio(value, reading)
    except AudioLookupError:
        recording = None
    if recording is not None:
        return recording
    return generate_audio(value)


def generate_audio(value: str) -> AudioDownload | None:
    text = strip_furigana(value)
    if not text:
        return None
    try:
        data = asyncio.run(_synthesize(text))
        validate_audio(data)
    except AudioLookupError:
        raise
    except Exception as exc:
        raise AudioLookupError(
            "Не удалось создать озвучку. Проверьте интернет и повторите попытку."
        ) from exc
    return AudioDownload(
        data, ".mp3", "https://github.com/rany2/edge-tts", "",
        "Microsoft Edge · Nanami", "Синтез речи · Nanami",
    )


async def _synthesize(text: str) -> bytes:
    audio = bytearray()
    async with asyncio.timeout(60):
        async for chunk in edge_tts.Communicate(text, VOICE).stream():
            if chunk["type"] != "audio":
                continue
            audio.extend(chunk["data"])
            if len(audio) > MAX_AUDIO_BYTES:
                raise AudioLookupError("Аудиофайл больше 10 МБ.")
    if not audio:
        raise AudioLookupError("Сервис озвучки не вернул аудио. Повторите попытку.")
    return bytes(audio)
