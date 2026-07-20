from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .jobs import JobManager
from .review_store import ReviewStore
from .routes import router


def create_app(*, root: Path | None = None, manager: JobManager | None = None) -> FastAPI:
    package = Path(__file__).parent
    app = FastAPI(title="anki-pipeline", docs_url=None, redoc_url=None)
    app.state.root = (root or Path.cwd()).resolve()
    app.state.jobs = manager or JobManager(output_root=app.state.root / "output")
    app.state.reviews = ReviewStore()
    app.mount("/static", StaticFiles(directory=package / "static"), name="static")
    app.include_router(router)
    return app
