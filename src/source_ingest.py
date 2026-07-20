from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

from .semantic import save_json_atomic

YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be",
}
FILE_MARKER = "__ANKI_PIPELINE_FILE__"


class SourceIngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedSource:
    kind: str
    video_path: str
    source_url: str
    source_id: str
    title: str
    reused: bool
    prepared_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def prepare_local_source(path: Path) -> PreparedSource:
    if not path.exists():
        raise SourceIngestError(f"Локальный видеофайл не найден: {path}")
    if not path.is_file():
        raise SourceIngestError(f"Указанный путь не является файлом: {path}")
    resolved = path.resolve()
    return PreparedSource(
        kind="local", video_path=str(resolved), source_url="",
        source_id=hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16],
        title=path.stem, reused=True, prepared_at="",
    )


def normalize_youtube_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
    except ValueError as exc:
        raise SourceIngestError(f"Некорректная YouTube-ссылка: {value}") from exc
    host = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or host not in YOUTUBE_HOSTS:
        raise SourceIngestError(
            "Разрешены только http/https-ссылки youtube.com или youtu.be."
        )
    return urlunsplit(("https", parsed.netloc.casefold(), parsed.path, parsed.query, ""))


def _load_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _url_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _read_metadata(video_path: Path) -> tuple[str, str]:
    info_path = video_path.with_suffix(".info.json")
    if not info_path.exists():
        return video_path.parent.name, video_path.stem
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
        return str(data.get("id") or video_path.parent.name), str(
            data.get("title") or video_path.stem
        )
    except (OSError, json.JSONDecodeError):
        return video_path.parent.name, video_path.stem


def prepare_youtube_source(
    url: str, *, input_root: Path = Path("input/downloads"),
    subtitle_languages: tuple[str, ...] = (), force: bool = False,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> PreparedSource:
    normalized_url = normalize_youtube_url(url)
    input_root.mkdir(parents=True, exist_ok=True)
    index_path = input_root / "source_index.json"
    index = _load_index(index_path)
    key = _url_key(normalized_url)
    cached = index.get(key)
    if cached and not force:
        cached_path = Path(str(cached.get("video_path", "")))
        if cached_path.is_file() and _is_within(cached_path, input_root):
            return PreparedSource(
                kind="youtube", video_path=str(cached_path),
                source_url=normalized_url, source_id=str(cached.get("source_id", "")),
                title=str(cached.get("title", cached_path.stem)), reused=True,
                prepared_at=str(cached.get("prepared_at", "")),
            )
    if importlib.util.find_spec("yt_dlp") is None:
        raise SourceIngestError(
            "yt-dlp не установлен. Выполните: py -m pip install -r requirements.txt"
        )
    output_template = str(
        (input_root.resolve() / "%(extractor_key)s" / "%(id)s" / "video.%(ext)s")
    )
    command = [
        sys.executable, "-m", "yt_dlp", "--no-playlist", "--newline", "--progress",
        "--no-remote-components", "--windows-filenames", "--trim-filenames", "120",
        "--format", "bv*+ba/b", "--merge-output-format", "mkv",
        "--write-info-json", "--no-write-playlist-metafiles",
        "--print", f"after_move:{FILE_MARKER}%(filepath)s",
        "--output", output_template,
    ]
    languages = [value.strip() for value in subtitle_languages if value.strip()]
    if languages:
        command.extend([
            "--write-subs", "--write-auto-subs", "--sub-format", "vtt/best",
            "--sub-langs", ",".join(f"{value}.*" for value in dict.fromkeys(languages)),
        ])
    command.append(normalized_url)
    try:
        process = popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        raise SourceIngestError(f"Не удалось запустить yt-dlp: {exc}") from exc
    downloaded_path: Path | None = None
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        if clean.startswith(FILE_MARKER):
            downloaded_path = Path(clean[len(FILE_MARKER):].strip())
        elif clean:
            print(clean, flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise SourceIngestError(f"yt-dlp завершился с кодом {return_code}.")
    if downloaded_path is None or not downloaded_path.is_file():
        raise SourceIngestError("yt-dlp не сообщил путь к загруженному видео.")
    if not _is_within(downloaded_path, input_root):
        raise SourceIngestError("yt-dlp сообщил путь за пределами каталога загрузок.")
    source_id, title = _read_metadata(downloaded_path)
    prepared_at = datetime.now(timezone.utc).isoformat()
    prepared = PreparedSource(
        kind="youtube", video_path=str(downloaded_path.resolve()),
        source_url=normalized_url, source_id=source_id, title=title,
        reused=False, prepared_at=prepared_at,
    )
    manifest_path = downloaded_path.parent / "source_manifest.json"
    save_json_atomic(prepared.to_dict(), manifest_path)
    index[key] = prepared.to_dict()
    save_json_atomic(index, index_path)
    return prepared


def prepare_source(
    source: str | Path, *, input_root: Path = Path("input/downloads"),
    subtitle_languages: tuple[str, ...] = (), force: bool = False,
) -> PreparedSource:
    value = str(source).strip()
    if value.lower().startswith(("http://", "https://")):
        return prepare_youtube_source(
            value, input_root=input_root,
            subtitle_languages=subtitle_languages, force=force,
        )
    return prepare_local_source(Path(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Подготовить локальный видеоисточник для anki-pipeline"
    )
    parser.add_argument("source", help="YouTube URL или путь к локальному видео")
    parser.add_argument("--language", default="en", help="Язык основных субтитров")
    parser.add_argument("--target-language", help="Изучаемый язык субтитров")
    parser.add_argument("--force", action="store_true", help="Скачать URL повторно")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        languages = tuple(
            value for value in (args.language, args.target_language) if value
        )
        prepared = prepare_source(
            args.source, subtitle_languages=languages, force=args.force
        )
    except SourceIngestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(prepared.video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
