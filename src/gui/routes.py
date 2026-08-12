from __future__ import annotations

import asyncio
import html
import json
import os
import re
import secrets
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from ..paths import build_result_paths
from ..notebooklm import NotebookLMImportError, get_notebooklm_prompt, import_notebooklm_cards
from ..source_ingest import SourceIngestError, prepare_source
from .review_store import ReviewError, normalize_uploaded_image
from .schemas import PipelineRunRequest, STAGES, STAGE_LABELS
from .services import (
    compute_plan,
    export_approved_deck,
    generate_image_candidates,
    migrate_legacy_project,
)

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}
MAX_UPLOAD = 8 * 1024 * 1024 * 1024


def _csrf(request: Request) -> str:
    token = request.cookies.get("anki_csrf")
    return token if token and len(token) == 43 else secrets.token_urlsafe(32)


def _check_csrf(request: Request, token: str) -> None:
    cookie = request.cookies.get("anki_csrf")
    if not cookie or not secrets.compare_digest(cookie, token):
        raise HTTPException(403, "Недействительный токен формы.")


def _render(request: Request, name: str, context: dict[str, Any], status: int = 200) -> HTMLResponse:
    token = _csrf(request)
    response = templates.TemplateResponse(request=request, name=name, context={**context, "csrf": token}, status_code=status)
    response.set_cookie("anki_csrf", token, httponly=True, samesite="strict")
    return response


def _job(request: Request, job_id: str):
    try:
        return request.app.state.jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Задача не найдена.") from exc


def _safe_output(state: Any, root: Path) -> Path:
    output = Path(state.output_directory).resolve()
    allowed = (root / "output").resolve()
    if not output.is_relative_to(allowed):
        raise HTTPException(403, "Недопустимый проект.")
    return output


def _alternatives(output: Path, card_id: str) -> list[dict[str, Any]]:
    candidate_file = output / "media_candidates" / card_id / "candidates.json"
    try:
        value = json.loads(candidate_file.read_text(encoding="utf-8")).get("candidates", [])
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _media_fragment(request: Request, state: Any, output: Path, card_id: str) -> HTMLResponse:
    review = request.app.state.reviews.load_or_create(output)
    card = next((item for item in review["cards"] if item["card_id"] == card_id), None)
    if card is None:
        raise HTTPException(404, "Карточка не найдена.")
    return _render(request, "media_controls.html", {
        "job": state,
        "card": {**card, "alternatives": _alternatives(output, card_id)},
    })


def _export_fragment(
    request: Request,
    state: Any,
    output: Path,
    review: dict[str, Any],
    *,
    deck_name: str = "",
    global_tags: str = "",
    exported: bool = False,
) -> HTMLResponse:
    included = sum(bool(card.get("included")) for card in review["cards"])
    return _render(request, "export_panel.html", {
        "job": state,
        "review": review,
        "included": included,
        "excluded": len(review["cards"]) - included,
        "deck_name": deck_name or state.source,
        "global_tags": global_tags,
        "exported": exported,
    })


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    root = request.app.state.root
    projects = []
    output_root = root / "output"
    known_jobs = getattr(request.app.state.jobs, "jobs", {})
    jobs_by_output = {
        Path(state.output_directory).resolve(): state
        for state in known_jobs.values()
    }
    for directory in output_root.iterdir() if output_root.exists() else []:
        if not directory.is_dir() or not any((directory / name).exists() for name in ("transcript.json", "card_candidates.json", "approved_cards.json")):
            continue
        review = None
        try:
            review = json.loads((directory / "approved_cards.json").read_text(encoding="utf-8")) if (directory / "approved_cards.json").exists() else None
        except (OSError, json.JSONDecodeError):
            pass
        state = jobs_by_output.get(directory.resolve())
        projects.append({
            "name": directory.name,
            "modified": directory.stat().st_mtime,
            "review": review and review.get("status"),
            "deck": any(directory.glob("*.apkg")),
            "url": (
                f"/jobs/{state.job_id}/review"
                if state and (directory / "card_candidates.json").is_file()
                else f"/jobs/{state.job_id}" if state
                else f"/projects/{directory.name}"
            ),
        })
    projects.sort(key=lambda project: project["modified"], reverse=True)
    return _render(request, "index.html", {
        "stages": STAGE_LABELS, "projects": projects,
        "gemini_key_detected": bool(os.getenv("GEMINI_API_KEY")),
    })


async def _save_upload(upload: UploadFile, root: Path) -> Path:
    original = Path(upload.filename or "").name
    suffix = Path(original).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError("Разрешены только MP4, MKV, WebM, MOV, M4V и AVI.")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original).stem).strip("-")[:60] or "video"
    directory = root / "input" / "uploads" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / f"{stem}{suffix}"
    fd, temporary = tempfile.mkstemp(prefix=".upload-", dir=directory)
    size = 0
    try:
        with os.fdopen(fd, "wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise ValueError("Видео превышает лимит 8 ГБ.")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        await upload.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


@router.post("/jobs")
async def create_job(
    request: Request, csrf: Annotated[str, Form()], youtube_url: Annotated[str, Form()] = "",
    video_file: Annotated[UploadFile | None, File()] = None,
    stages: Annotated[list[str] | None, Form()] = None,
    model: Annotated[str, Form()] = "small", device: Annotated[str, Form()] = "auto",
    language: Annotated[str, Form()] = "", target_language: Annotated[str, Form()] = "ja",
    ollama_model: Annotated[str, Form()] = "qwen3:8b", ollama_url: Annotated[str, Form()] = "http://localhost:11434",
    ollama_timeout: Annotated[float, Form()] = 120.0, frames_per_card: Annotated[int, Form()] = 5,
    llm_provider: Annotated[str, Form()] = "ollama",
    gemini_model: Annotated[str, Form()] = "gemini-3.1-flash-lite",
    knowledge_language: Annotated[str, Form()] = "en",
    image_policy: Annotated[str, Form()] = "auto", force: Annotated[bool, Form()] = False,
):
    _check_csrf(request, csrf)
    has_file = bool(video_file and video_file.filename)
    if bool(youtube_url.strip()) == has_file:
        return _render(request, "error.html", {"message": "Укажите ровно один источник: YouTube URL или видеофайл."}, 422)
    try:
        if has_file:
            video = await _save_upload(video_file, request.app.state.root)  # type: ignore[arg-type]
            prepared = prepare_source(video)
        else:
            prepared = prepare_source(youtube_url.strip())
        selected = list(stages or [])
        if not selected:
            raise ValueError("Выберите хотя бы одну стадию.")
        if knowledge_language not in {"en", "ru", "source"}:
            raise ValueError("Неизвестный язык карточек.")
        video_path = Path(prepared.video_path)
        migrate_legacy_project(video_path, request.app.state.root / "output")
        paths = build_result_paths(video_path)
        plan = compute_plan(selected, paths.directory)
        job = request.app.state.jobs.submit(PipelineRunRequest(
            video_path=video_path, source=prepared.title, stages=plan,
            model=model, device=device, language=language or None, force=force,
            ollama_model=ollama_model, ollama_url=ollama_url, ollama_timeout=ollama_timeout,
            llm_provider=llm_provider, gemini_model=gemini_model,
            knowledge_language=knowledge_language,
            target_language=target_language, frames_per_card=frames_per_card, image_policy=image_policy,
        ))
    except (ValueError, SourceIngestError) as exc:
        return _render(request, "error.html", {"message": str(exc)}, 422)
    return RedirectResponse(f"/jobs/{job.job_id}", 303)


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    if (output / "card_candidates.json").is_file() and state.status not in {"queued", "running"}:
        return RedirectResponse(f"/jobs/{job_id}/review", 303)
    return _render(request, "job.html", {
        "job": state,
        "labels": STAGE_LABELS,
        "notebooklm_prompt": get_notebooklm_prompt(),
    })


@router.get("/projects/{project_name}")
def open_project(request: Request, project_name: str):
    output_root = (request.app.state.root / "output").resolve()
    output = (output_root / project_name).resolve()
    if (
        Path(project_name).name != project_name
        or not output.is_relative_to(output_root)
        or not output.is_dir()
    ):
        raise HTTPException(404, "Проект не найден.")
    for state in getattr(request.app.state.jobs, "jobs", {}).values():
        if Path(state.output_directory).resolve() == output:
            destination = "review" if (output / "card_candidates.json").is_file() else ""
            suffix = f"/{destination}" if destination else ""
            return RedirectResponse(f"/jobs/{state.job_id}{suffix}", 303)

    source = project_name
    video = request.app.state.root / "input" / f"{project_name}.mp4"
    try:
        transcript = json.loads((output / "transcript.json").read_text(encoding="utf-8"))
        source = str(transcript.get("source") or project_name)
        candidate = Path(source)
        if candidate.is_file():
            video = candidate
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    state = request.app.state.jobs.register_existing(video, output)
    state.source = Path(source).name or project_name
    has_cards = (output / "card_candidates.json").is_file()
    if not has_cards:
        state.status = "completed"
        state.completed_stages = ["transcription"]
        state.progress_message = "Транскрипция готова"
    destination = "review" if has_cards else ""
    suffix = f"/{destination}" if destination else ""
    return RedirectResponse(f"/jobs/{state.job_id}{suffix}", 303)


@router.get("/jobs/{job_id}/transcript")
def download_transcript(request: Request, job_id: str):
    state = _job(request, job_id)
    path = _safe_output(state, request.app.state.root) / "transcript.txt"
    if not path.is_file():
        raise HTTPException(404, "Транскрипция ещё не готова.")
    return FileResponse(path, filename=f"{Path(state.video_path).stem}-transcript.txt")


@router.get("/jobs/{job_id}/notebooklm-prompt")
def download_notebooklm_prompt(request: Request, job_id: str):
    _job(request, job_id)
    return PlainTextResponse(get_notebooklm_prompt())


@router.post("/jobs/{job_id}/notebooklm-cards")
async def import_cards_from_notebooklm(
    request: Request,
    job_id: str,
    csrf: Annotated[str, Form()],
    cards_file: Annotated[UploadFile | None, File()] = None,
    cards_text: Annotated[str, Form()] = "",
    frames_per_card: Annotated[int, Form()] = 5,
    image_policy: Annotated[str, Form()] = "auto",
    append: Annotated[bool, Form()] = False,
):
    _check_csrf(request, csrf)
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    pasted_text = cards_text.strip()
    has_file = bool(cards_file and cards_file.filename)
    if has_file and pasted_text:
        return _render(request, "error.html", {"message": "Вставьте текст или загрузите файл — не оба сразу."}, 422)
    if not has_file and not pasted_text:
        return _render(request, "error.html", {"message": "Вставьте ответ NotebookLM или загрузите .txt файл."}, 422)
    if has_file:
        assert cards_file is not None
        if Path(cards_file.filename or "").suffix.lower() != ".txt":
            await cards_file.close()
            return _render(request, "error.html", {"message": "Загрузите .txt файл из NotebookLM."}, 422)
        data = await cards_file.read(2 * 1024 * 1024 + 1)
        await cards_file.close()
    else:
        data = pasted_text.encode("utf-8")
    if len(data) > 2 * 1024 * 1024:
        return _render(request, "error.html", {"message": "Файл карточек больше 2 МБ."}, 422)
    try:
        import_notebooklm_cards(
            data.decode("utf-8"),
            output / "transcript.json",
            output / "card_candidates.json",
            append=append,
        )
        if not 1 <= frames_per_card <= 20 or image_policy not in {"auto", "always", "never"}:
            raise ValueError("Некорректные параметры подбора кадров.")
        job = request.app.state.jobs.submit(PipelineRunRequest(
            video_path=Path(state.video_path), source=state.source, stages=["frame_extraction"],
            frames_per_card=frames_per_card, image_policy=image_policy, force=True,
        ))
    except (UnicodeDecodeError, NotebookLMImportError, ValueError) as exc:
        return _render(request, "error.html", {"message": str(exc)}, 422)
    return RedirectResponse(f"/jobs/{job.job_id}", 303)


@router.get("/jobs/{job_id}/events")
async def events(request: Request, job_id: str):
    async def stream():
        last = None
        while True:
            state = _job(request, job_id)
            payload = json.dumps(request.app.state.jobs.public(state), ensure_ascii=False)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if state.status in {"awaiting_review", "completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(1)
    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/jobs/{job_id}/review", response_class=HTMLResponse)
def review_page(request: Request, job_id: str):
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    try:
        review = request.app.state.reviews.load_or_create(output)
        automatic = json.loads((output / "card_candidates.json").read_text(encoding="utf-8"))
    except (ReviewError, OSError, json.JSONDecodeError) as exc:
        return _render(request, "error.html", {"message": str(exc)}, 422)
    raw = {item["card_id"]: item for item in automatic["card_candidates"]}
    cards = []
    for card in review["cards"]:
        cards.append({
            **card,
            "meta": raw.get(card["card_id"], {}),
            "alternatives": _alternatives(output, card["card_id"]),
        })
    summary = {"total": len(cards), "included": sum(c["included"] for c in cards), "reviewed": sum(c["reviewed"] for c in cards), "no_image": sum((c.get("image") or {}).get("mode") == "none" for c in cards)}
    return _render(request, "review.html", {"job": state, "review": review, "cards": cards, "summary": summary})


@router.post("/jobs/{job_id}/review/cards/{card_id}", response_class=HTMLResponse)
def save_card(
    request: Request, job_id: str, card_id: str, csrf: Annotated[str, Form()],
    included: Annotated[bool, Form()] = False, front: Annotated[str, Form()] = "",
    back: Annotated[str, Form()] = "", extra: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "", difficulty: Annotated[int, Form()] = 1,
    importance: Annotated[int, Form()] = 1,
):
    _check_csrf(request, csrf)
    state = _job(request, job_id)
    try:
        card = request.app.state.reviews.save_card(_safe_output(state, request.app.state.root), card_id, {
            "included": included, "front": front, "back": back, "extra": extra,
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
            "difficulty": difficulty, "importance": importance,
        })
    except ReviewError as exc:
        return HTMLResponse(f'<span class="error">{str(exc)}</span>', 422)
    return HTMLResponse('<span class="saved">Сохранено · review снова draft</span>')


@router.post("/jobs/{job_id}/review/cards/{card_id}/image-remove")
def remove_image(request: Request, job_id: str, card_id: str, csrf: Annotated[str, Form()]):
    _check_csrf(request, csrf)
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    request.app.state.reviews.save_card(output, card_id, {"image": {"mode": "none"}})
    return _media_fragment(request, state, output, card_id)


@router.post("/jobs/{job_id}/review/cards/{card_id}/image-candidates")
def image_candidates(request: Request, job_id: str, card_id: str, csrf: Annotated[str, Form()]):
    _check_csrf(request, csrf)
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    try:
        generate_image_candidates(Path(state.video_path), output, card_id)
    except Exception as exc:
        return _render(request, "error.html", {"message": str(exc)}, 422)
    return _media_fragment(request, state, output, card_id)


@router.post("/jobs/{job_id}/review/cards/{card_id}/image-select")
def select_image(request: Request, job_id: str, card_id: str, csrf: Annotated[str, Form()], path: Annotated[str, Form()], timestamp: Annotated[float, Form()] = 0.0):
    _check_csrf(request, csrf)
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    candidate = Path(path)
    expected = (output / "media_candidates" / card_id).resolve()
    resolved = (output / candidate).resolve()
    if candidate.is_absolute() or not resolved.is_relative_to(expected) or not resolved.is_file():
        raise HTTPException(422, "Недопустимый кадр.")
    request.app.state.reviews.save_card(output, card_id, {"image": {"mode": "selected_candidate", "path": candidate.as_posix(), "timestamp": timestamp}})
    return _media_fragment(request, state, output, card_id)


@router.post("/jobs/{job_id}/review/cards/{card_id}/image-upload")
async def upload_image(request: Request, job_id: str, card_id: str, csrf: Annotated[str, Form()], image: Annotated[UploadFile, File()]):
    _check_csrf(request, csrf)
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    data = await image.read(10 * 1024 * 1024 + 1)
    destination = output / "review_media" / card_id / "uploaded.jpg"
    try:
        normalize_uploaded_image(data, destination)
        relative = destination.relative_to(output).as_posix()
        request.app.state.reviews.save_card(output, card_id, {"image": {"mode": "uploaded", "path": relative, "timestamp": 0.0}})
    except ReviewError as exc:
        return _render(request, "error.html", {"message": str(exc)}, 422)
    return _media_fragment(request, state, output, card_id)


@router.post("/jobs/{job_id}/review/approve")
def approve(request: Request, job_id: str, csrf: Annotated[str, Form()]):
    _check_csrf(request, csrf)
    state = _job(request, job_id)
    try:
        output = _safe_output(state, request.app.state.root)
        review = request.app.state.reviews.approve(output)
    except ReviewError as exc:
        return _render(request, "error.html", {"message": str(exc)}, 422)
    return _export_fragment(request, state, output, review)


@router.post("/jobs/{job_id}/export")
def export(
    request: Request,
    job_id: str,
    csrf: Annotated[str, Form()],
    deck_name: Annotated[str, Form()] = "",
    global_tags: Annotated[str, Form()] = "",
):
    _check_csrf(request, csrf)
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    clean_name = " ".join(deck_name.split())
    tags = list(dict.fromkeys(
        tag.strip() for tag in re.split(r"[,\s]+", global_tags) if tag.strip()
    ))
    if not clean_name or len(clean_name) > 120:
        return HTMLResponse('<p class="error">Название колоды должно содержать от 1 до 120 символов.</p>', 422)
    if any(len(tag) > 60 for tag in tags):
        return HTMLResponse('<p class="error">Каждый общий тег должен быть не длиннее 60 символов.</p>', 422)
    try:
        deck = export_approved_deck(Path(state.video_path), clean_name, tags)
        review = request.app.state.reviews.load_or_create(output)
    except Exception as exc:
        return HTMLResponse(f'<p class="error">{html.escape(str(exc))}</p>', 422)
    state.status = "completed"
    return _export_fragment(
        request, state, output, review, deck_name=clean_name,
        global_tags=", ".join(tags), exported=True,
    )


@router.get("/jobs/{job_id}/deck")
def deck(request: Request, job_id: str):
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    matches = list(output.glob("*.apkg"))
    if len(matches) != 1:
        raise HTTPException(404, "Колода не найдена.")
    return FileResponse(matches[0], filename=matches[0].name)


@router.get("/jobs/{job_id}/media/{card_id}")
def media(request: Request, job_id: str, card_id: str):
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    review = request.app.state.reviews.load_or_create(output)
    card = next((item for item in review["cards"] if item["card_id"] == card_id), None)
    image = card and card.get("image")
    if not image or image.get("mode") == "none":
        raise HTTPException(404)
    path = (output / image["path"]).resolve()
    if not path.is_relative_to(output) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@router.get("/jobs/{job_id}/review/cards/{card_id}/image-candidates/{candidate_index}")
def candidate_image(
    request: Request, job_id: str, card_id: str, candidate_index: int
):
    state = _job(request, job_id)
    output = _safe_output(state, request.app.state.root)
    manifest = output / "media_candidates" / card_id / "candidates.json"
    try:
        candidates = json.loads(manifest.read_text(encoding="utf-8"))["candidates"]
        candidate = candidates[candidate_index]
        relative = Path(candidate["path"])
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        raise HTTPException(404, "Кадр не найден.")
    expected = (output / "media_candidates" / card_id).resolve()
    path = (output / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(expected) or not path.is_file():
        raise HTTPException(404, "Кадр не найден.")
    return FileResponse(path)


@router.get("/jobs/{job_id}/video")
def video(request: Request, job_id: str):
    state = _job(request, job_id)
    path = Path(state.video_path).resolve()
    roots = [(request.app.state.root / name).resolve() for name in ("input",)]
    if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
        raise HTTPException(403)
    return FileResponse(path)
