from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..anki_export import AnkiExportError, build_anki_deck
from ..project_store import ProjectError


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
MAX_CARDS_UPLOAD = 2 * 1024 * 1024
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
    project_name: Annotated[str, Form()],
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
    return _render(
        request,
        "review.html",
        {"project": project, "included": included, "deck_ready": deck_ready},
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
def update_card(
    request: Request,
    project_id: str,
    card_id: str,
    csrf: Annotated[str, Form()],
    front: Annotated[str, Form()] = "",
    back: Annotated[str, Form()] = "",
    extra: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "",
    included: Annotated[bool, Form()] = False,
):
    _check_csrf(request, csrf)
    parsed_tags = [tag for tag in re.split(r"[,\s]+", tags) if tag]
    try:
        request.app.state.projects.update_card(
            project_id,
            card_id,
            {"front": front, "back": back, "extra": extra, "tags": parsed_tags, "included": included},
        )
    except ProjectError as exc:
        return _error(request, str(exc))
    return RedirectResponse(f"/projects/{project_id}#{card_id}", 303)


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
