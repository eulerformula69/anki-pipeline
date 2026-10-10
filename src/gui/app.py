from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..project_store import ProjectStore
from .routes import router


def create_app(root: Path | None = None) -> FastAPI:
    project_root = (root or Path.cwd()).resolve()
    app = FastAPI(title="anki-pipeline")
    app.state.root = project_root
    app.state.projects = ProjectStore(project_root / "output")
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    app.include_router(router)
    return app
