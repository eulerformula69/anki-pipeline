from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.audio import AudioExtractionError, ensure_ffmpeg, extract_audio
    from src.anki_export import AnkiExportError, build_anki_deck
    from src.cards import CardGenerationError, generate_card_candidates, localize_existing_candidates
    from src.frame_extraction import FrameExtractionError, extract_card_images
    from src.knowledge import KnowledgeExtractionError, extract_knowledge
    from src.output import save_transcript_json, save_transcript_text
    from src.ollama_client import OllamaClient, OllamaError
    from src.paths import build_result_paths
    from src.review import KnowledgeReviewError, review_knowledge
    from src.semantic import SemanticSegmentationError, semantic_segment_transcript
    from src.transcription import TranscriptionError, resolve_device, transcribe_audio
    from src.target_expressions import TargetExpressionError, extract_target_expressions
else:
    from .audio import AudioExtractionError, ensure_ffmpeg, extract_audio
    from .anki_export import AnkiExportError, build_anki_deck
    from .cards import CardGenerationError, generate_card_candidates, localize_existing_candidates
    from .frame_extraction import FrameExtractionError, extract_card_images
    from .knowledge import KnowledgeExtractionError, extract_knowledge
    from .output import save_transcript_json, save_transcript_text
    from .ollama_client import OllamaClient, OllamaError
    from .paths import build_result_paths
    from .review import KnowledgeReviewError, review_knowledge
    from .semantic import SemanticSegmentationError, semantic_segment_transcript
    from .transcription import TranscriptionError, resolve_device, transcribe_audio
    from .target_expressions import TargetExpressionError, extract_target_expressions

LOGGER = logging.getLogger("anki_pipeline")


def validate_input(video: Path) -> None:
    if not video.exists():
        raise FileNotFoundError(f"Входной файл не найден: {video}")
    if not video.is_file():
        raise ValueError(f"Указанный путь не является файлом: {video}")


def run_pipeline(
    video: Path,
    *,
    model: str = "small",
    device: str = "auto",
    language: str | None = None,
    force: bool = False,
    semantic_segment: bool = False,
    ollama_model: str = "qwen3:8b",
    ollama_url: str = "http://localhost:11434",
    ollama_timeout: float = 120.0,
    extract_knowledge_items: bool = False,
    knowledge_language: str = "source",
    knowledge_min_importance: int = 1,
    knowledge_max_items_per_segment: int = 12,
    review_knowledge_items: bool = False,
    generate_cards: bool = False,
    extract_expressions: bool = False,
    target_language: str = "ja",
    target_glossary: Path | None = None,
    extract_card_images_stage: bool = False,
    frames_per_card: int = 5,
    image_policy: str = "auto",
    build_anki_deck_stage: bool = False,
    deck_name: str | None = None,
) -> Path:
    paths = build_result_paths(video)
    if sum((extract_knowledge_items, review_knowledge_items, generate_cards, extract_expressions, extract_card_images_stage, build_anki_deck_stage)) > 1:
        raise ValueError(
            "Используйте только один из флагов --extract-knowledge, "
            "--review-knowledge, --extract-target-expressions или "
            "--generate-card-candidates, --extract-card-images или --build-anki-deck за запуск."
        )

    if build_anki_deck_stage:
        if paths.anki_deck.exists() and paths.anki_export_manifest_json.exists() and not force:
            LOGGER.info("Колода Anki уже существует: %s", paths.anki_deck)
            return paths.anki_deck
        if not paths.card_candidates_json.exists():
            raise AnkiExportError(
                "Не найден card_candidates.json. Сначала выполните --generate-card-candidates."
            )
        build_anki_deck(
            paths.card_candidates_json, paths.target_expressions_json,
            paths.card_media_json, paths.anki_deck,
            paths.anki_export_manifest_json, deck_name=deck_name,
        )
        LOGGER.info("Колода Anki: %s", paths.anki_deck)
        return paths.anki_deck

    if extract_card_images_stage:
        if paths.card_media_json.exists() and not force:
            LOGGER.info("Медиа карточек уже существует: %s", paths.card_media_json)
            return paths.card_media_json
        if not paths.transcript_json.exists():
            raise FrameExtractionError(
                "Не найден transcript.json. Сначала запустите транскрипцию видео."
            )
        if not paths.card_candidates_json.exists():
            raise FrameExtractionError(
                "Не найден card_candidates.json. Сначала выполните --generate-card-candidates."
            )
        extract_card_images(
            video, paths.transcript_json, paths.card_candidates_json,
            paths.card_media_json, paths.media_directory,
            frames_per_card=frames_per_card, image_policy=image_policy,
        )
        LOGGER.info("Медиа карточек: %s", paths.card_media_json)
        return paths.card_media_json

    if extract_expressions:
        if paths.target_expressions_json.exists() and not force:
            LOGGER.info("Выражения уже существуют: %s", paths.target_expressions_json)
            return paths.target_expressions_json
        if not paths.knowledge_items_json.exists():
            raise TargetExpressionError(
                "Не найден knowledge_items.json. Сначала выполните --extract-knowledge."
            )
        LOGGER.info("Извлечение выражений языка %s: model=%s", target_language, ollama_model)
        client = OllamaClient(ollama_model, ollama_url, ollama_timeout)
        extract_target_expressions(
            paths.transcript_json,
            paths.knowledge_items_json,
            paths.target_expressions_json,
            paths.target_expressions_diagnostic_json,
            client,
            target_language=target_language,
            glossary_path=target_glossary or Path("glossaries") / f"{target_language}.json",
        )
        LOGGER.info("Целевые выражения: %s", paths.target_expressions_json)
        return paths.target_expressions_json

    if generate_cards:
        if paths.card_candidates_json.exists() and not force:
            if paths.target_expressions_json.exists():
                count = localize_existing_candidates(
                    paths.card_candidates_json, paths.target_expressions_json
                )
                LOGGER.info("Локализовано карточек: %d", count)
            LOGGER.info("Кандидаты карточек уже существуют: %s", paths.card_candidates_json)
            return paths.card_candidates_json
        if not paths.knowledge_items_json.exists():
            raise CardGenerationError(
                "Не найден knowledge_items.json. Сначала выполните --extract-knowledge."
            )
        if not paths.knowledge_review_json.exists():
            raise CardGenerationError(
                "Не найден knowledge_review.json. Сначала выполните --review-knowledge."
            )
        LOGGER.info("Генерация кандидатов карточек: model=%s", ollama_model)
        client = OllamaClient(ollama_model, ollama_url, ollama_timeout)
        generate_card_candidates(
            paths.knowledge_items_json,
            paths.knowledge_review_json,
            paths.card_candidates_json,
            paths.cards_diagnostic_json,
            client,
            paths.target_expressions_json,
        )
        LOGGER.info("Кандидаты карточек: %s", paths.card_candidates_json)
        return paths.card_candidates_json

    if review_knowledge_items:
        if paths.knowledge_review_json.exists() and not force:
            LOGGER.info("Knowledge Review уже существует: %s", paths.knowledge_review_json)
            return paths.knowledge_review_json
        if not paths.knowledge_items_json.exists():
            raise KnowledgeReviewError(
                "Не найден knowledge_items.json. Сначала выполните --extract-knowledge."
            )
        LOGGER.info("Knowledge Review через Ollama: model=%s", ollama_model)
        client = OllamaClient(ollama_model, ollama_url, ollama_timeout)
        review_knowledge(
            paths.knowledge_items_json,
            paths.knowledge_review_json,
            paths.review_diagnostic_json,
            client,
        )
        LOGGER.info("Knowledge Review: %s", paths.knowledge_review_json)
        return paths.knowledge_review_json

    if extract_knowledge_items:
        if paths.knowledge_items_json.exists() and not force:
            LOGGER.info("Учебные знания уже существуют: %s", paths.knowledge_items_json)
            return paths.knowledge_items_json
        if not paths.transcript_json.exists():
            raise KnowledgeExtractionError(
                "Не найден transcript.json. Сначала запустите транскрибацию видео."
            )
        if not paths.semantic_segments_json.exists():
            raise KnowledgeExtractionError(
                "Не найден semantic_segments.json. Сначала выполните: "
                f'py -m src.pipeline "{video}" --semantic-segment'
            )
        LOGGER.info("Извлечение атомарных знаний: model=%s", ollama_model)
        client = OllamaClient(
            model=ollama_model,
            base_url=ollama_url,
            timeout=ollama_timeout,
        )
        extract_knowledge(
            paths.transcript_json,
            paths.semantic_segments_json,
            paths.knowledge_items_json,
            paths.knowledge_diagnostic_json,
            client,
            language=knowledge_language,
            min_importance=knowledge_min_importance,
            max_items_per_segment=knowledge_max_items_per_segment,
        )
        LOGGER.info("Учебные знания: %s", paths.knowledge_items_json)
        LOGGER.info("Готово: %s", paths.directory)
        return paths.knowledge_items_json

    validate_input(video)

    if paths.transcript_json.exists() and paths.transcript_txt.exists() and not force:
        LOGGER.info("Транскрипция уже существует: %s", paths.transcript_json)
    else:
        ensure_ffmpeg()
        if paths.audio.exists() and not force:
            LOGGER.info("Используется существующее аудио: %s", paths.audio)
        else:
            LOGGER.info("Извлечение mono-аудио 16 kHz: %s", paths.audio)
            extract_audio(video, paths.audio, force=force)

        selected_device, compute_type = resolve_device(device)
        LOGGER.info(
            "Транскрибация: model=%s device=%s compute=%s",
            model,
            selected_device,
            compute_type,
        )
        transcript = transcribe_audio(
            paths.audio,
            video,
            model_name=model,
            device=device,
            language=language,
        )
        save_transcript_json(transcript, paths.transcript_json)
        save_transcript_text(transcript, paths.transcript_txt)

    if semantic_segment:
        if paths.semantic_segments_json.exists() and not force:
            LOGGER.info(
                "Смысловая сегментация уже существует: %s",
                paths.semantic_segments_json,
            )
        else:
            LOGGER.info("Смысловая сегментация через Ollama: model=%s", ollama_model)
            client = OllamaClient(
                model=ollama_model,
                base_url=ollama_url,
                timeout=ollama_timeout,
            )
            semantic_segment_transcript(
                paths.transcript_json,
                paths.semantic_segments_json,
                paths.semantic_diagnostic_json,
                client,
            )
            LOGGER.info("Смысловые блоки: %s", paths.semantic_segments_json)

    LOGGER.info("Готово: %s", paths.directory)
    return paths.transcript_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Локальная транскрибация видео")
    parser.add_argument("video", type=Path, help="Путь к видеофайлу")
    parser.add_argument("--model", default="small", help="Модель Whisper (default: small)")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="Устройство"
    )
    parser.add_argument("--language", help="Код языка, например ru; по умолчанию авто")
    parser.add_argument("--force", action="store_true", help="Перезаписать результаты")
    parser.add_argument(
        "--semantic-segment",
        action="store_true",
        help="Создать смысловые блоки через локальный Ollama",
    )
    parser.add_argument(
        "--ollama-model", default="qwen3:8b", help="Модель Ollama (default: qwen3:8b)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="URL локального API Ollama",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=120.0,
        help="Таймаут одного запроса к Ollama в секундах",
    )
    parser.add_argument(
        "--extract-knowledge",
        action="store_true",
        help="Извлечь атомарные знания из готовых смысловых блоков",
    )
    parser.add_argument(
        "--knowledge-language",
        choices=("source", "ru", "en"),
        default="source",
        help="Язык тезисов и пояснений (default: source)",
    )
    parser.add_argument(
        "--knowledge-min-importance",
        type=int,
        choices=range(1, 6),
        default=1,
        metavar="1-5",
        help="Минимальная importance после проверки (default: 1)",
    )
    parser.add_argument(
        "--knowledge-max-items-per-segment",
        type=int,
        default=12,
        help="Максимум знаний из одного смыслового блока (default: 12)",
    )
    parser.add_argument(
        "--review-knowledge",
        action="store_true",
        help="Оценить качество готовых knowledge items",
    )
    parser.add_argument(
        "--generate-card-candidates",
        action="store_true",
        help="Создать кандидаты будущих карточек",
    )
    parser.add_argument(
        "--extract-target-expressions",
        action="store_true",
        help="Найти выражения изучаемого языка и их оригинальное написание",
    )
    parser.add_argument(
        "--target-language",
        default="ja",
        help="Код изучаемого языка (default: ja)",
    )
    parser.add_argument(
        "--target-glossary",
        type=Path,
        help="Локальный JSON-глоссарий; по умолчанию glossaries/<language>.json",
    )
    parser.add_argument(
        "--extract-card-images", action="store_true",
        help="Локально выбрать кадры для готовых кандидатов карточек",
    )
    parser.add_argument(
        "--frames-per-card", type=int, choices=range(1, 21), default=5, metavar="1-20",
        help="Число кадров-кандидатов для карточки (default: 5)",
    )
    parser.add_argument(
        "--image-policy", choices=("auto", "always", "never"), default="auto",
        help="Политика выбора изображения (default: auto)",
    )
    parser.add_argument(
        "--build-anki-deck", action="store_true",
        help="Собрать готовую локальную колоду .apkg",
    )
    parser.add_argument(
        "--deck-name", help="Имя колоды Anki; по умолчанию имя исходного видео",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    try:
        run_pipeline(
            args.video,
            model=args.model,
            device=args.device,
            language=args.language,
            force=args.force,
            semantic_segment=args.semantic_segment,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            ollama_timeout=args.ollama_timeout,
            extract_knowledge_items=args.extract_knowledge,
            knowledge_language=args.knowledge_language,
            knowledge_min_importance=args.knowledge_min_importance,
            knowledge_max_items_per_segment=args.knowledge_max_items_per_segment,
            review_knowledge_items=args.review_knowledge,
            generate_cards=args.generate_card_candidates,
            extract_expressions=args.extract_target_expressions,
            target_language=args.target_language,
            target_glossary=args.target_glossary,
            extract_card_images_stage=args.extract_card_images,
            frames_per_card=args.frames_per_card,
            image_policy=args.image_policy,
            build_anki_deck_stage=args.build_anki_deck,
            deck_name=args.deck_name,
        )
    except (
        FileNotFoundError,
        ValueError,
        AudioExtractionError,
        TranscriptionError,
        OllamaError,
        SemanticSegmentationError,
        KnowledgeExtractionError,
        KnowledgeReviewError,
        CardGenerationError,
        TargetExpressionError,
        FrameExtractionError,
        AnkiExportError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
