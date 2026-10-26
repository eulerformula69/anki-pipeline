from __future__ import annotations

import hashlib
import html
import json
import re
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from mutagen import File as AudioFile
from mutagen import MutagenError


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
JPOD101_AUDIO = "https://assets.languagepod101.com/dictionary/japanese/audiomp3.php"
JPOD101_MISSING_MD5 = "7e2c2f954ef6051373ba916f000168dc"
USER_AGENT = "anki-pipeline/1.0 (personal language-learning tool)"
MAX_AUDIO_BYTES = 10 * 1024 * 1024
JAPANESE_CHARS = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々〆ヵヶ"
FURIGANA = re.compile(
    rf"(?P<base>[{JAPANESE_CHARS}]+)\[(?P<reading>[\u3040-\u30ffー・]+)\]"
)
JAPANESE_RUN = re.compile(
    rf"[{JAPANESE_CHARS}](?:[{JAPANESE_CHARS}\s、。！？!?…・〜～「」『』（）()ー]*"
    rf"[{JAPANESE_CHARS}。！？!?])?"
)


class AudioLookupError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioDownload:
    data: bytes
    suffix: str
    source_url: str
    license_name: str
    attribution: str
    source_name: str = "Wikimedia Commons"


def validate_audio(data: bytes) -> None:
    try:
        audio = AudioFile(BytesIO(data))
        if audio is not None and audio.info.length >= 0.15:
            return
    except (MutagenError, ValueError, EOFError):
        pass
    raise AudioLookupError("Аудиофайл повреждён или слишком короткий. Выберите другую запись.")


def strip_furigana(value: str) -> str:
    result = FURIGANA.sub(lambda match: match.group("base"), value)
    result = re.sub(rf"(?<=[{JAPANESE_CHARS}])\s+(?=[{JAPANESE_CHARS}])", "", result)
    return result.strip()


def furigana_reading(value: str) -> str:
    result = FURIGANA.sub(lambda match: match.group("reading"), value)
    result = re.sub(rf"(?<=[{JAPANESE_CHARS}])\s+(?=[{JAPANESE_CHARS}])", "", result)
    return result.strip()


def infer_audio_text(front: str, back: str) -> str:
    for field in (back, front):
        cleaned = strip_furigana(field)
        matches = [match.group(0).strip() for match in JAPANESE_RUN.finditer(cleaned)]
        if matches:
            return max(matches, key=lambda item: len(re.findall(rf"[{JAPANESE_CHARS}]", item)))
    return ""


def infer_audio_reading(front: str, back: str) -> str:
    for field in (back, front):
        reading = furigana_reading(field)
        matches = [match.group(0).strip() for match in JAPANESE_RUN.finditer(reading)]
        if matches:
            return max(matches, key=lambda item: len(re.findall(rf"[{JAPANESE_CHARS}]", item)))
    return ""


def forvo_search_url(value: str) -> str:
    return f"https://forvo.com/word/{quote(value, safe='')}/#ja"


def find_native_audio(value: str, reading: str) -> AudioDownload | None:
    jpod101_error = None
    try:
        jpod101 = find_jpod101_audio(value, reading)
    except AudioLookupError as exc:
        jpod101 = None
        jpod101_error = exc
    if jpod101 is not None:
        return jpod101
    japanese_length = len(re.findall(rf"[{JAPANESE_CHARS}]", value))
    if japanese_length > 8:
        return None
    try:
        return find_commons_audio(value)
    except AudioLookupError:
        if jpod101_error is not None:
            raise jpod101_error
        return None


def find_jpod101_audio(value: str, reading: str) -> AudioDownload | None:
    target = strip_furigana(value)
    clean_reading = furigana_reading(reading)
    if not target or not clean_reading or len(target) > 40:
        return None
    url = f"{JPOD101_AUDIO}?{urlencode({'kanji': target, 'kana': clean_reading})}"
    data = _request_bytes(url)
    digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
    if digest == JPOD101_MISSING_MD5:
        return None
    validate_audio(data)
    return AudioDownload(
        data,
        ".mp3",
        "https://www.japanesepod101.com/japanese-dictionary/",
        "",
        "JapanesePod101",
        "JapanesePod101",
    )


def find_commons_audio(value: str) -> AudioDownload | None:
    target = strip_furigana(value)
    if not target or len(target) > 40 or not re.search(rf"[{JAPANESE_CHARS}]", target):
        return None
    search = _request_json({
        "action": "query",
        "list": "search",
        "srsearch": f'"「{target}」" filetype:audio',
        "srnamespace": "6",
        "srlimit": "10",
        "format": "json",
        "formatversion": "2",
    })
    titles = [
        item["title"]
        for item in search.get("query", {}).get("search", [])
        if isinstance(item.get("title"), str)
    ]
    return _commons_file(titles, target)


def _commons_file(titles: list[str], target: str) -> AudioDownload | None:
    if not titles:
        return None
    result = _request_json({
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "format": "json",
        "formatversion": "2",
    })
    pages = result.get("query", {}).get("pages", [])
    for page in pages:
        info_items = page.get("imageinfo", [])
        if not info_items:
            continue
        info = info_items[0]
        metadata = info.get("extmetadata", {})
        description = _metadata_text(metadata, "ImageDescription")
        categories = _metadata_text(metadata, "Categories")
        if target not in description:
            continue
        if "Japanese" not in description and "Japanese pronunciation" not in categories:
            continue
        license_name = _metadata_text(metadata, "LicenseShortName")
        if not _open_license(license_name):
            continue
        audio_url = info.get("url")
        source_url = info.get("descriptionurl")
        if not isinstance(audio_url, str) or not isinstance(source_url, str):
            continue
        suffix = Path(urlparse(audio_url).path).suffix.casefold()
        if suffix not in {".mp3", ".ogg", ".oga", ".wav"}:
            continue
        data = _request_bytes(audio_url)
        try:
            validate_audio(data)
        except AudioLookupError:
            continue
        attribution = _metadata_text(metadata, "Artist") or "Wikimedia Commons contributor"
        return AudioDownload(data, suffix, source_url, license_name, attribution)
    return None


def _open_license(value: str) -> bool:
    normalized = value.casefold()
    return "creative commons" in normalized or normalized.startswith("cc ") or "public domain" in normalized


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    raw = metadata.get(key, {})
    value = raw.get("value", "") if isinstance(raw, dict) else ""
    if not isinstance(value, str):
        return ""
    without_tags = re.sub(r"<[^>]+>", "", value)
    return " ".join(html.unescape(without_tags).split())


def _request_json(parameters: dict[str, str]) -> dict[str, Any]:
    data = _request_bytes(f"{COMMONS_API}?{urlencode(parameters)}")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AudioLookupError("Wikimedia Commons вернул некорректный ответ.") from exc
    if not isinstance(value, dict):
        raise AudioLookupError("Wikimedia Commons вернул некорректный ответ.")
    return value


def _request_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=15) as response:
            size = response.headers.get("Content-Length")
            if size and int(size) > MAX_AUDIO_BYTES:
                raise AudioLookupError("Аудиофайл больше 10 МБ.")
            data = response.read(MAX_AUDIO_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise AudioLookupError(f"Не удалось получить аудио: {exc}") from exc
    if len(data) > MAX_AUDIO_BYTES:
        raise AudioLookupError("Аудиофайл больше 10 МБ.")
    return data
