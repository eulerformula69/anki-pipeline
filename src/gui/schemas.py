from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


STAGES = (
    "transcription", "target_expressions", "target_transcription",
    "semantic_segmentation", "knowledge_extraction", "knowledge_review",
    "card_generation", "frame_extraction",
)

STAGE_LABELS = {
    "transcription": "Транскрипция основного аудио",
    "target_expressions": "Выражения изучаемого языка",
    "target_transcription": "Второй проход транскрипции",
    "semantic_segmentation": "Смысловая сегментация",
    "knowledge_extraction": "Извлечение знаний",
    "knowledge_review": "Knowledge Review",
    "card_generation": "Кандидаты карточек",
    "frame_extraction": "Изображения для карточек",
}


@dataclass
class PipelineRunRequest:
    video_path: Path
    stages: list[str]
    source: str
    model: str = "small"
    device: str = "auto"
    language: str | None = None
    force: bool = False
    ollama_model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout: float = 120.0
    llm_provider: str = "ollama"
    gemini_model: str = "gemini-3.1-flash-lite"
    knowledge_language: str = "en"
    knowledge_min_importance: int = 1
    knowledge_max_items_per_segment: int = 12
    target_language: str = "ja"
    target_glossary: Path | None = None
    target_whisper_model: str = "small"
    frames_per_card: int = 5
    image_policy: str = "auto"


@dataclass
class JobState:
    job_id: str
    source: str
    video_path: str
    output_directory: str
    status: str = "queued"
    current_stage: str | None = None
    stages: list[str] = field(default_factory=list)
    completed_stages: list[str] = field(default_factory=list)
    progress_message: str = "Ожидает запуска"
    error: str | None = None
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
