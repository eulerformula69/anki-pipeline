from __future__ import annotations

import json
import hashlib
import logging
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TargetExpression, TargetExpressionCollection, TranscriptSegment
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


def build_prompt(
    segments: list[TranscriptSegment], language: str, correction: str = "",
    native_evidence: dict[int, list[str]] | None = None,
) -> str:
    window_ids = {segment.id for segment in segments}
    target_groups: dict[str, list[int]] = {}
    for source_id in window_ids:
        for text in (native_evidence or {}).get(source_id, []):
            target_groups.setdefault(text, []).append(source_id)
    payload = {
        "primary_segments": [segment.__dict__ for segment in segments],
        "target_language_segments": [
            {"text": text, "source_transcript_segment_ids": sorted(set(source_ids))}
            for text, source_ids in target_groups.items()
        ],
    }
    return f"""Extract the useful target-language material from this mixed-language transcript.
Target language code: {language}.
Return isolated words, grammar constructions, and complete example sentences.
type must be term, grammar_pattern, or example_sentence.
The transcript may contain inaccurate romanization. Reconstruct native writing cautiously.
target_language_transcript is a noisy second Whisper pass over the same audio interval.
Use its native spelling only when it phonetically agrees with romanized target-language
speech in the primary text. Never extract an expression from the second pass alone.
The explanations may be in another language. Use that language in meaning.
surface must use the target language's native writing system (for Japanese: kana/kanji).
reading is the native pronunciation aid (for Japanese: kana). romanization is Latin script.
Include an example_sentence when the speaker gives a target-language example, not only its words.
Return at most 6 items. Prioritize complete example sentences, then grammar patterns,
then important vocabulary. Do not list every particle separately.
Do not turn ordinary English explanation into Japanese. Return each expression once per window.
source_transcript_segment_ids must come only from INPUT. Do not return knowledge_ids.
For Japanese, surface and reading MUST contain Japanese characters. Examples:
surface="だって", reading="だって", romanization="datte". Never put romaji in surface.
surface="明日は晴れだって", reading="あしたははれだって", romanization="ashita wa hare datte".
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


def recover_partial_expressions(raw_response: str) -> dict[str, list[dict[str, Any]]] | None:
    match = re.search(r'"expressions"\s*:\s*\[', raw_response)
    if match is None:
        return None
    decoder = json.JSONDecoder()
    position = match.end()
    recovered: list[dict[str, Any]] = []
    while position < len(raw_response):
        while position < len(raw_response) and raw_response[position] in " \r\n\t,":
            position += 1
        if position >= len(raw_response) or raw_response[position] != "{":
            break
        try:
            value, position = decoder.raw_decode(raw_response, position)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            recovered.append(value)
    return {"expressions": recovered} if recovered else None


def parse_response(
    data: dict[str, Any], items: list, language: str,
    glossary: dict[str, dict[str, str]] | None = None,
    allowed_source_ids: set[int] | None = None,
    native_evidence: dict[int, list[str]] | None = None,
    primary_evidence: dict[int, str] | None = None,
) -> list[TargetExpression]:
    raw_items = data.get("expressions")
    if set(data) != {"expressions"} or not isinstance(raw_items, list):
        raise TargetExpressionError("Ответ должен содержать только массив expressions.")
    knowledge = {item.id: item for item in items}
    allowed_segments = allowed_source_ids or {
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
            "type", "surface", "reading", "romanization", "meaning",
            "source_transcript_segment_ids", "confidence",
        }
        if set(raw) != required:
            errors.append(f"Expression {index}: неверный набор полей.")
            continue
        strings = [raw.get(key) for key in ("surface", "reading", "romanization", "meaning")]
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
            or (language == "ja" and re.search(r"[A-Za-z]", raw["surface"] + raw["reading"]))
        ):
            errors.append(
                f"Expression {index}: surface и reading должны использовать письменность {language}."
            )
            continue
        expression_type = (
            str(glossary_entry.get("type"))
            if glossary_entry and glossary_entry.get("type")
            else raw.get("type")
        )
        if expression_type not in {"term", "grammar_pattern", "example_sentence"}:
            errors.append(f"Expression {index}: неизвестный type.")
            continue
        if (
            not isinstance(source_ids, list) or not source_ids
            or any(not isinstance(value, int) or value not in allowed_segments for value in source_ids)
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
        evidence_text = " ".join(
            text for source_id in source_ids
            for text in (native_evidence or {}).get(source_id, [])
        )
        transcribed = (
            bool(evidence_text)
            and _normalize_native(raw["surface"]) in _normalize_native(evidence_text)
            and romanization_matches_primary(
                raw["romanization"], " ".join(
                    (primary_evidence or {}).get(source_id, "") for source_id in source_ids
                )
            )
        )
        result.append(TargetExpression(
            id=f"e_{len(result) + 1:04d}", language=language,
            surface=raw["surface"].strip(), reading=raw["reading"].strip(),
            romanization=raw["romanization"].strip(), meaning=raw["meaning"].strip(),
            knowledge_ids=[
                item.id for item in items
                if set(item.source_transcript_segment_ids) & set(source_ids)
            ],
            source_transcript_segment_ids=sorted(set(source_ids)),
            verification=(
                "verified" if glossary_entry
                else "transcribed" if transcribed
                else "needs_review"
            ),
            confidence=float(confidence),
            type=expression_type,
        ))
    if errors and not result:
        raise TargetExpressionError(" | ".join(errors))
    if errors:
        LOGGER.warning(
            "Пропущено некорректных выражений: %d; корректные сохранены.", len(errors)
        )
    return result


def _normalize_native(text: str) -> str:
    return re.sub(r"[\s、。,.!?！？'\"「」『』（）()]+", "", text).casefold()


def stable_expression_id(language: str, expression_type: str, surface: str) -> str:
    key = f"{language}\0{expression_type}\0{_normalize_native(surface)}".encode("utf-8")
    return "e_" + hashlib.blake2s(key, digest_size=6).hexdigest()


def romanization_matches_primary(romanization: str, primary: str) -> bool:
    roman_tokens = re.findall(r"[a-z]+", romanization.casefold())
    primary_tokens = re.findall(r"[a-z]+", primary.casefold())
    if not roman_tokens or not primary_tokens:
        return False
    needle = "".join(roman_tokens)
    if any(
        primary_tokens[start:start + len(roman_tokens)] == roman_tokens
        for start in range(0, len(primary_tokens) - len(roman_tokens) + 1)
    ):
        return True
    minimum = max(1, len(roman_tokens) - 1)
    maximum = min(len(primary_tokens), len(roman_tokens) + 1)
    return any(
        SequenceMatcher(None, needle, "".join(primary_tokens[start:start + size])).ratio()
        >= 0.82
        for size in range(minimum, maximum + 1)
        for start in range(0, len(primary_tokens) - size + 1)
    )


def load_native_evidence(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        evidence: dict[int, list[str]] = {}
        for segment in data["segments"]:
            text = str(segment["text"]).strip()
            for source_id in segment["source_transcript_segment_ids"]:
                evidence.setdefault(int(source_id), []).append(text)
        return evidence
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TargetExpressionError(f"Не удалось прочитать target_transcript.json: {exc}") from exc


def extract_target_expressions(
    transcript_path: Path, knowledge_path: Path, destination: Path, diagnostic_path: Path,
    client: OllamaClient, *, target_language: str,
    glossary_path: Path | None = None,
    target_transcript_path: Path | None = None,
) -> TargetExpressionCollection:
    if isinstance(client, OllamaClient):
        client.format_schema = SCHEMA
        client.num_predict = 1400
    source, _, items = load_knowledge_items(knowledge_path)
    try:
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript_segments = [TranscriptSegment(**item) for item in transcript_data["segments"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TargetExpressionError(f"Не удалось прочитать transcript.json: {exc}") from exc
    glossary = load_glossary(glossary_path)
    native_evidence = load_native_evidence(target_transcript_path)
    primary_evidence = {segment.id: segment.text for segment in transcript_segments}
    all_expressions: list[TargetExpression] = []
    window_size, overlap = 12, 2
    starts = range(0, len(transcript_segments), window_size - overlap)
    windows = [transcript_segments[start:start + window_size] for start in starts]
    for window_number, window in enumerate(windows, start=1):
        if not window:
            continue
        responses: list[str] = []
        error = ""
        for attempt in range(2):
            correction = "" if not attempt else (
                f"\nVALIDATION ERROR: {error}. Correct the complete JSON now."
            )
            try:
                label = f"Ollama Expressions: окно {window_number}/{len(windows)}"
                with ActivityIndicator(label + (" (retry)" if attempt else "")):
                    data, raw = client.generate_json(build_prompt(
                        window, target_language, correction, native_evidence
                    ))
                responses.append(raw)
                all_expressions.extend(parse_response(
                    data, items, target_language, glossary,
                    {segment.id for segment in window},
                    native_evidence,
                    primary_evidence,
                ))
                break
            except OllamaInvalidResponseError as exc:
                responses.append(exc.raw_response)
                error = str(exc)
                partial = recover_partial_expressions(exc.raw_response)
                if partial is not None:
                    try:
                        recovered = parse_response(
                            partial, items, target_language, glossary,
                            {segment.id for segment in window},
                            native_evidence, primary_evidence,
                        )
                    except TargetExpressionError:
                        pass
                    else:
                        all_expressions.extend(recovered)
                        LOGGER.warning(
                            "Окно %d/%d: сохранено выражений из оборванного ответа: %d.",
                            window_number, len(windows), len(recovered),
                        )
                        break
            except TargetExpressionError as exc:
                error = str(exc)
            except OllamaError:
                raise
            if attempt == 1:
                if error and all(
                    "письменность" in part
                    for part in error.split(" | ")
                ):
                    LOGGER.warning(
                        "Окно %d/%d не содержит валидных выражений %s; пропущено.",
                        window_number,
                        len(windows),
                        target_language,
                    )
                    break
                save_json_atomic({
                    "window": window_number, "error": error,
                    "ollama_responses": responses, "model": client.model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }, diagnostic_path)
                raise TargetExpressionError(
                    f"Ollama дважды вернул некорректные выражения. Диагностика: {diagnostic_path}"
                )
    merged: dict[tuple[str, str], TargetExpression] = {}
    for expression in all_expressions:
        key = (expression.type, expression.surface)
        previous = merged.get(key)
        if previous is None:
            merged[key] = expression
        else:
            merged[key] = TargetExpression(
                **{**previous.__dict__,
                   "knowledge_ids": list(dict.fromkeys(previous.knowledge_ids + expression.knowledge_ids)),
                   "source_transcript_segment_ids": sorted(set(previous.source_transcript_segment_ids + expression.source_transcript_segment_ids)),
                   "confidence": max(previous.confidence, expression.confidence)}
            )
    expressions = [
        TargetExpression(**{
            **item.__dict__,
            "id": stable_expression_id(item.language, item.type, item.surface),
        })
        for item in merged.values()
    ]
    result = TargetExpressionCollection(source, client.model, target_language, expressions)
    save_json_atomic(result.to_dict(), destination)
    diagnostic_path.unlink(missing_ok=True)
    return result


def load_target_expressions(path: Path) -> list[TargetExpression]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [TargetExpression(**item) for item in data["expressions"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TargetExpressionError(f"Не удалось прочитать target_expressions.json: {exc}") from exc
