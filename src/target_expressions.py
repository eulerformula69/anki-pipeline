from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TargetExpression, TargetExpressionCollection
from .ollama_client import OllamaClient, OllamaError, OllamaInvalidResponseError
from .ollama_schemas import target_expression_response_schema
from .progress import ActivityIndicator
from .review import load_knowledge_items
from .semantic import save_json_atomic

LOGGER = logging.getLogger("anki_pipeline.target_expressions")
SCHEMA = target_expression_response_schema()
NATIVE_SCRIPT_PATTERNS = {
    "ja": re.compile(r"[\u3040-\u30ff\u3400-\u9fff]"),
    "ko": re.compile(r"[\uac00-\ud7af]"),
    "zh": re.compile(r"[\u3400-\u9fff]"),
}


class TargetExpressionError(RuntimeError):
    pass


def build_prompt(items: list, language: str, correction: str = "") -> str:
    payload = [item.to_dict() for item in items]
    return f"""Find words, phrases, and grammar patterns in the TARGET language.
Target language code: {language}.
The explanations may be in another language. Preserve that language in meaning.
surface must use the target language's native writing system (for Japanese: kana/kanji).
reading is the native pronunciation aid (for Japanese: kana). romanization is Latin script.
Do not invent an expression when the evidence is ambiguous. Return each expression once.
For Japanese, surface and reading MUST contain Japanese characters; for example
surface="だって", reading="だって", romanization="datte". Never put romaji in surface.
knowledge_ids and source_transcript_segment_ids must come only from INPUT.
confidence is confidence that surface exactly matches the spoken romanized expression.
Return strict JSON only.{correction}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}"""


def load_glossary(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["entries"]
        return {str(item["romanization"]).casefold(): item for item in entries}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TargetExpressionError(f"Не удалось прочитать глоссарий {path}: {exc}") from exc


def parse_response(
    data: dict[str, Any], items: list, language: str,
    glossary: dict[str, dict[str, str]] | None = None,
) -> list[TargetExpression]:
    raw_items = data.get("expressions")
    if set(data) != {"expressions"} or not isinstance(raw_items, list):
        raise TargetExpressionError("Ответ должен содержать только массив expressions.")
    knowledge = {item.id: item for item in items}
    allowed_segments = {
        segment_id for item in items for segment_id in item.source_transcript_segment_ids
    }
    seen: set[tuple[str, str]] = set()
    result: list[TargetExpression] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            errors.append(f"Expression {index}: ожидается объект.")
            continue
        required = {
            "surface", "reading", "romanization", "meaning", "knowledge_ids",
            "source_transcript_segment_ids", "confidence",
        }
        if set(raw) != required:
            errors.append(f"Expression {index}: неверный набор полей.")
            continue
        strings = [raw.get(key) for key in ("surface", "reading", "romanization", "meaning")]
        ids = raw.get("knowledge_ids")
        source_ids = raw.get("source_transcript_segment_ids")
        confidence = raw.get("confidence")
        if not all(isinstance(value, str) and value.strip() for value in strings):
            errors.append(f"Expression {index}: текстовые поля не должны быть пустыми.")
            continue
        glossary_entry = (glossary or {}).get(raw["romanization"].strip().casefold())
        if glossary_entry:
            raw = {
                **raw,
                "surface": str(glossary_entry["surface"]),
                "reading": str(glossary_entry.get("reading", glossary_entry["surface"])),
            }
        native_pattern = NATIVE_SCRIPT_PATTERNS.get(language)
        if native_pattern and (
            not native_pattern.search(raw["surface"])
            or not native_pattern.search(raw["reading"])
        ):
            errors.append(
                f"Expression {index}: surface и reading должны использовать письменность {language}."
            )
            continue
        if not isinstance(ids, list) or not ids or any(item not in knowledge for item in ids):
            errors.append(f"Expression {index}: неизвестные или пустые knowledge_ids.")
            continue
        expected_ids = {
            segment_id for item_id in ids
            for segment_id in knowledge[item_id].source_transcript_segment_ids
        }
        if (
            not isinstance(source_ids, list) or not source_ids
            or any(not isinstance(value, int) or value not in allowed_segments for value in source_ids)
            or not set(source_ids).issubset(expected_ids)
        ):
            errors.append(f"Expression {index}: неверные source_transcript_segment_ids.")
            continue
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            errors.append(f"Expression {index}: confidence должен быть от 0 до 1.")
            continue
        key = (language, raw["surface"].strip())
        if key in seen:
            continue
        seen.add(key)
        result.append(TargetExpression(
            id=f"e_{len(result) + 1:04d}", language=language,
            surface=raw["surface"].strip(), reading=raw["reading"].strip(),
            romanization=raw["romanization"].strip(), meaning=raw["meaning"].strip(),
            knowledge_ids=list(dict.fromkeys(ids)), source_transcript_segment_ids=sorted(set(source_ids)),
            verification="verified" if glossary_entry else "needs_review",
            confidence=float(confidence),
        ))
    if errors:
        raise TargetExpressionError(" | ".join(errors))
    return result


def extract_target_expressions(
    knowledge_path: Path, destination: Path, diagnostic_path: Path,
    client: OllamaClient, *, target_language: str,
    glossary_path: Path | None = None,
) -> TargetExpressionCollection:
    if isinstance(client, OllamaClient):
        client.format_schema = SCHEMA
    source, _, items = load_knowledge_items(knowledge_path)
    glossary = load_glossary(glossary_path)
    responses: list[str] = []
    error = ""
    for attempt in range(2):
        correction = ""
        if attempt:
            correction = f"\nPrevious response failed validation: {error}. Fix the complete response."
        try:
            with ActivityIndicator("Ollama Expressions" + (" (retry)" if attempt else "")):
                data, raw = client.generate_json(build_prompt(items, target_language, correction))
            responses.append(raw)
            expressions = parse_response(data, items, target_language, glossary)
            result = TargetExpressionCollection(source, client.model, target_language, expressions)
            save_json_atomic(result.to_dict(), destination)
            diagnostic_path.unlink(missing_ok=True)
            return result
        except OllamaInvalidResponseError as exc:
            responses.append(exc.raw_response)
            error = str(exc)
        except TargetExpressionError as exc:
            error = str(exc)
        except OllamaError:
            raise
    save_json_atomic({
        "error": error, "ollama_responses": responses, "model": client.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, diagnostic_path)
    raise TargetExpressionError(
        f"Ollama дважды вернул некорректные выражения. Диагностика: {diagnostic_path}"
    )


def load_target_expressions(path: Path) -> list[TargetExpression]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [TargetExpression(**item) for item in data["expressions"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TargetExpressionError(f"Не удалось прочитать target_expressions.json: {exc}") from exc
