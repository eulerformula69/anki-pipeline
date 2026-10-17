from __future__ import annotations

import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..audio_sources import AudioDownload, AudioLookupError, forvo_search_url
from ..audio_tts import find_or_generate_audio
from ..anki_export import AnkiExportError, build_anki_deck
from ..project_store import ProjectError


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
MAX_CARDS_UPLOAD = 2 * 1024 * 1024
MAX_AUDIO_UPLOAD = 10 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".mp3", ".ogg", ".oga", ".wav"}
PROMPT_PATH = Path(__file__).parents[2] / "notebooklm-prompt.md"


def _csrf(request: Request) -> str:
    token = request.cookies.get("anki_csrf")
    return token if token and len(token) == 43 else secrets.token_urlsafe(32)


def _check_csrf(request: Request, token: str) -> None:
    cookie = request.cookies.get("anki_csrf")
    if not cookie or not secrets.compare_digest(cookie, token):
        raise HTTPException(403, "Недействительный токен формы.")


def _render(
    request: Request, template: str, context: dict[str, Any], status: int = 200
) -> HTMLResponse:
    token = _csrf(request)
    response = templates.TemplateResponse(
        request=request,
        name=template,
        context={**context, "csrf": token},
        status_code=status,
    )
    response.set_cookie("anki_csrf", token, httponly=True, samesite="strict")
    return response


def _error(request: Request, message: str, status: int = 422) -> HTMLResponse:
    return _render(request, "error.html", {"message": message}, status)


async def _cards_text(cards_text: str, cards_file: UploadFile | None) -> str:
    pasted = cards_text.strip()
    has_file = bool(cards_file and cards_file.filename)
    if has_file == bool(pasted):
        raise ProjectError("Вставьте ответ LLM или загрузите один .txt-файл.")
    if not has_file:
        return pasted
    assert cards_file is not None
    if Path(cards_file.filename or "").suffix.casefold() != ".txt":
        await cards_file.close()
        raise ProjectError("Поддерживаются только .txt-файлы.")
    data = await cards_file.read(MAX_CARDS_UPLOAD + 1)
    await cards_file.close()
    if len(data) > MAX_CARDS_UPLOAD:
        raise ProjectError("Файл больше 2 МБ.")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectError("Файл должен быть в кодировке UTF-8.") from exc


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render(request, "index.html", {"projects": request.app.state.projects.list()})


@router.get("/prompt", response_class=PlainTextResponse)
def prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


@router.post("/projects")
async def create_project(
    request: Request,
    csrf: Annotated[str, Form()],
    project_name: Annotated[str, Form()] = "",
    cards_text: Annotated[str, Form()] = "",
    cards_file: Annotated[UploadFile | None, File()] = None,
):
    _check_csrf(request, csrf)
    try:
        text = await _cards_text(cards_text, cards_file)
        project = request.app.state.projects.create(project_name, text)
    except ProjectError as exc:
        return _error(request, str(exc))
    return RedirectResponse(f"/projects/{project['id']}", 303)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def review(request: Request, project_id: str) -> HTMLResponse:
    try:
        project = request.app.state.projects.load(project_id)
    except ProjectError as exc:
        return _error(request, str(exc), 404)
    included = sum(card["included"] for card in project["cards"])
    deck_ready = (request.app.state.projects.root / project_id / "deck.apkg").is_file()
    for card in project["cards"]:
        card["forvo_url"] = forvo_search_url(card["audio_text"]) if card["audio_text"] else ""
    return _render(
        request,
        "review.html",
        {
            "project": project,
            "included": included,
            "deck_ready": deck_ready,
            "audio_found": request.query_params.get("audio_found"),
            "audio_missing": request.query_params.get("audio_missing"),
            "audio_failed": request.query_params.get("audio_failed"),
            "audio_uploaded": request.query_params.get("audio_uploaded"),
            "deck_created": request.query_params.get("exported"),
            "anki_added": request.query_params.get("anki_added"),
            "anki_updated": request.query_params.get("anki_updated"),
        },
    )


@router.post("/projects/{project_id}/cards")
async def append_cards(
    request: Request,
    project_id: str,
    csrf: Annotated[str, Form()],
    cards_text: Annotated[str, Form()] = "",
    cards_file: Annotated[UploadFile | None, File()] = None,
):
    _check_csrf(request, csrf)
    try:
        text = await _cards_text(cards_text, cards_file)
        request.app.state.projects.append(project_id, text)
    except ProjectError as exc:
        return _error(request, str(exc))
    return RedirectResponse(f"/projects/{project_id}", 303)


@router.post("/projects/{project_id}/cards/new")
def add_card(request: Request, project_id: str, csrf: Annotated[str, Form()]):
    _check_csrf(request, csrf)
    try:
        card_id = request.app.state.projects.add_card(project_id)
    except ProjectError as exc:
        return _error(request, str(exc))
    return RedirectResponse(f"/projects/{project_id}#{card_id}", 303)


@router.post("/projects/{project_id}/cards/{card_id}")
async def update_card(
    request: Request,
    project_id: str,
    card_id: str,
    csrf: Annotated[str, Form()],
    front: Annotated[str, Form()] = "",
    back: Annotated[str, Form()] = "",
    extra: Annotated[str, Form()] = "",
    audio_text: Annotated[str, Form()] = "",
    audio_reading: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "",
    included: Annotated[bool, Form()] = False,
    audio_file: Annotated[UploadFile | None, File()] = None,
):
    _check_csrf(request, csrf)
    parsed_tags = [tag for tag in re.split(r"[,\s]+", tags) if tag]
    upload: tuple[bytes, str] | None = None
    if audio_file and audio_file.filename:
        suffix = Path(audio_file.filename).suffix.casefold()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            await audio_file.close()
            return _error(request, "Аудио должно быть в формате MP3, OGG, OGA или WAV.")
        data = await audio_file.read(MAX_AUDIO_UPLOAD + 1)
        await audio_file.close()
        if len(data) > MAX_AUDIO_UPLOAD:
            return _error(request, "Аудиофайл больше 10 МБ.")
        upload = (data, suffix)
    try:
        request.app.state.projects.update_card(
            project_id,
            card_id,
            {
                "front": front,
                "back": back,
                "extra": extra,
                "audio_text": audio_text,
                "audio_reading": audio_reading,
                "tags": parsed_tags,
                "included": included,
            },
        )
        if upload:
            data, suffix = upload
            request.app.state.projects.attach_audio(
                project_id,
                card_id,
                AudioDownload(data, suffix, "", "", ""),
                source="Загружено вручную",
            )
    except ProjectError as exc:
        return _error(request, str(exc))
    suffix = "?audio_uploaded=1" if upload else ""
    return RedirectResponse(f"/projects/{project_id}{suffix}#{card_id}", 303)


@router.post("/projects/{project_id}/audio/find")
def find_audio(request: Request, project_id: str, csrf: Annotated[str, Form()]):
    _check_csrf(request, csrf)
    try:
        project = request.app.state.projects.load(project_id)
    except ProjectError as exc:
        return _error(request, str(exc), 404)
    found = missing = failed = 0
    candidates = [
        card for card in project["cards"]
        if card["included"]
        and card["audio_text"]
        and (
            not card["audio"]
            or card["audio"].get("text", card["audio_text"]) != card["audio_text"]
            or card["audio"].get("reading", card["audio_reading"]) != card["audio_reading"]
        )
    ]

    def lookup(card: dict[str, Any]) -> AudioDownload | AudioLookupError | None:
        try:
            return find_or_generate_audio(card["audio_text"], card["audio_reading"])
        except AudioLookupError as exc:
            return exc

    workers = min(4, len(candidates)) or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloads = list(executor.map(lookup, candidates))
    for card, download in zip(candidates, downloads, strict=True):
        if isinstance(download, AudioLookupError):
            failed += 1
            continue
        if download is None:
            missing += 1
            continue
        try:
            request.app.state.projects.attach_audio(
                project_id, card["id"], download, source=download.source_name
            )
        except ProjectError:
            failed += 1
            continue
        found += 1
    query = urlencode({"audio_found": found, "audio_missing": missing, "audio_failed": failed})
    return RedirectResponse(f"/projects/{project_id}?{query}", 303)


@router.get("/projects/{project_id}/cards/{card_id}/audio")
def preview_audio(request: Request, project_id: str, card_id: str):
    try:
        path = request.app.state.projects.audio_path(project_id, card_id)
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path)


@router.post("/projects/{project_id}/cards/{card_id}/move")
def move_card(
    request: Request,
    project_id: str,
    card_id: str,
    csrf: Annotated[str, Form()],
    direction: Annotated[str, Form()],
):
    _check_csrf(request, csrf)
    try:
        request.app.state.projects.move_card(project_id, card_id, direction)
    except ProjectError as exc:
        return _error(request, str(exc))
    return RedirectResponse(f"/projects/{project_id}#{card_id}", 303)


@router.post("/projects/{project_id}/export")
def export_deck(
    request: Request,
    project_id: str,
    csrf: Annotated[str, Form()],
    deck_name: Annotated[str, Form()],
    global_tags: Annotated[str, Form()] = "",
):
    _check_csrf(request, csrf)
    clean_name = " ".join(deck_name.split())
    tags = [tag for tag in re.split(r"[,\s]+", global_tags) if tag]
    if not clean_name or len(clean_name) > 120:
        return _error(request, "Название колоды должно содержать от 1 до 120 символов.")
    try:
        project = request.app.state.projects.load(project_id)
        destination = request.app.state.projects.root / project_id / "deck.apkg"
        build_anki_deck(project, destination, deck_name=clean_name, global_tags=tags)
    except (ProjectError, AnkiExportError) as exc:
        return _error(request, str(exc))
    return RedirectResponse(f"/projects/{project_id}?exported=1", 303)


@router.get("/projects/{project_id}/deck")
def download_deck(request: Request, project_id: str):
    try:
        project = request.app.state.projects.load(project_id)
    except ProjectError as exc:
        return _error(request, str(exc), 404)
    path = request.app.state.projects.root / project_id / "deck.apkg"
    if not path.is_file():
        raise HTTPException(404, "Колода ещё не создана.")
    filename = f"{_safe_filename(project['name'])}.apkg"
    return FileResponse(path, filename=filename)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", value).strip(" .")
    return cleaned[:100] or "anki-deck"
