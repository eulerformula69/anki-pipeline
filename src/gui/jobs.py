from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..paths import build_result_paths
from .review_store import atomic_json
from .schemas import JobState, PipelineRunRequest
from .services import PipelineRunner


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JobManager:
    def __init__(
        self,
        runner: PipelineRunner | None = None,
        *,
        output_root: Path = Path("output"),
    ) -> None:
        self.runner = runner or PipelineRunner()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anki-gui")
        self.jobs: dict[str, JobState] = {}
        self._lock = threading.RLock()
        self.output_root = output_root.resolve()
        self.restore()

    def restore(self) -> None:
        if not self.output_root.exists():
            return
        for path in self.output_root.glob("*/gui_job.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                state = JobState(**data)
                output = Path(state.output_directory).resolve()
                if output != path.parent.resolve() or not output.is_relative_to(self.output_root):
                    continue
                if state.status in {"queued", "running", "exporting"}:
                    state.status = "failed"
                    state.error = "Приложение было перезапущено во время выполнения задачи. Запустите нужные стадии повторно."
                    state.progress_message = "Задача прервана перезапуском приложения"
                    state.finished_at = _now()
                    atomic_json(asdict(state), path)
                self.jobs[state.job_id] = state
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                logging.getLogger("anki_pipeline.gui").warning(
                    "Пропущено некорректное состояние job: %s", path
                )

    def submit(self, request: PipelineRunRequest) -> JobState:
        paths = build_result_paths(request.video_path)
        state = JobState(
            job_id=uuid.uuid4().hex, source=request.source,
            video_path=str(request.video_path.resolve()), output_directory=str(paths.directory.resolve()),
            stages=list(request.stages), created_at=_now(),
        )
        with self._lock:
            self.jobs[state.job_id] = state
            self._persist(state)
        self.executor.submit(self._run, state.job_id, request)
        return state

    def _run(self, job_id: str, request: PipelineRunRequest) -> None:
        state = self.get(job_id)
        self._update(state, status="running", started_at=_now(), progress_message="Конвейер запущен")
        try:
            self.runner.run(request, lambda stage, event: self._progress(state, stage, event))
            review_ready = (Path(state.output_directory) / "card_candidates.json").exists()
            self._update(
                state, status="awaiting_review" if review_ready else "completed",
                current_stage=None, finished_at=_now(),
                progress_message="Готово к ручной проверке" if review_ready else "Выбранные стадии завершены",
            )
        except Exception as exc:
            log_path = Path(state.output_directory) / "gui.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logging.getLogger("anki_pipeline.gui").exception("Job %s failed", job_id)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            self._update(state, status="failed", error=str(exc), finished_at=_now(), progress_message="Ошибка выполнения")

    def _progress(self, state: JobState, stage: str, event: str) -> None:
        if event == "started":
            self._update(state, current_stage=stage, progress_message=f"Выполняется: {stage}")
        else:
            if stage not in state.completed_stages:
                state.completed_stages.append(stage)
            self._update(state, progress_message=f"Завершено: {stage}")

    def _update(self, state: JobState, **values: object) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(state, key, value)
            self._persist(state)

    def _persist(self, state: JobState) -> None:
        atomic_json(asdict(state), Path(state.output_directory) / "gui_job.json")

    def get(self, job_id: str) -> JobState:
        with self._lock:
            try:
                return self.jobs[job_id]
            except KeyError as exc:
                raise KeyError("Задача не найдена.") from exc

    def register_existing(
        self, video: Path, output: Path, *, source: str | None = None,
        has_video: bool = True,
    ) -> JobState:
        state = JobState(
            job_id=uuid.uuid4().hex, source=source or video.name, video_path=str(video.resolve()),
            output_directory=str(output.resolve()), status="awaiting_review", created_at=_now(),
            has_video=has_video,
        )
        with self._lock:
            self.jobs[state.job_id] = state
            self._persist(state)
        return state

    def public(self, state: JobState) -> dict[str, object]:
        data = asdict(state)
        data.pop("video_path", None)
        data.pop("output_directory", None)
        return data
