from __future__ import annotations

import re
import threading
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse

from ..anki_connect import AnkiConnect, AnkiConnectError, load_settings, save_settings
from ..anki_export import AnkiExportError
from ..project_store import ProjectError
from .routes import _check_csrf, _error

router = APIRouter()
IMPORT_LOCK = threading.Lock()


@router.post("/anki-connect/decks")
def decks(request: Request, csrf: Annotated[str, Form()]):
    _check_csrf(request, csrf)
    settings = None
    try:
        settings = load_settings(request.app.state.projects.root)
        return {"decks": AnkiConnect(**settings).decks(), "url": settings["url"]}
    except AnkiConnectError as exc:
        return JSONResponse({"error": str(exc), "url": settings["url"] if settings else None}, status_code=422)


@router.post("/anki-connect/settings")
def settings(request: Request, csrf: Annotated[str, Form()],
             url: Annotated[str, Form()], key: Annotated[str, Form()] = ""):
    _check_csrf(request, csrf)
    try:
        connection = AnkiConnect(url, key)
        names = connection.decks()
        save_settings(request.app.state.projects.root, connection.url, key)
        return {"decks": names, "url": connection.url}
    except (AnkiConnectError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


@router.post("/projects/{project_id}/anki-connect")
def import_project(request: Request, project_id: str, csrf: Annotated[str, Form()],
                   deck_name: Annotated[str, Form()], global_tags: Annotated[str, Form()] = ""):
    _check_csrf(request, csrf)
    if not IMPORT_LOCK.acquire(blocking=False):
        return _error(request, "Дождитесь завершения текущего импорта.", 409)
    try:
        store = request.app.state.projects
        project = store.load(project_id)
        connection = AnkiConnect(**load_settings(store.root))
        tags = [tag for tag in re.split(r"[,\s]+", global_tags) if tag]
        added, updated = connection.import_project(project, store.root / project_id, deck_name, tags)
    except (AnkiConnectError, AnkiExportError, ProjectError) as exc:
        return _error(request, str(exc))
    finally:
        IMPORT_LOCK.release()
    return RedirectResponse(f"/projects/{project_id}?anki_added={added}&anki_updated={updated}", 303)
